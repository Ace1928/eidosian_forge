import json
import math
import os
import torch
import torch.nn.functional as F
from collections import defaultdict, Counter
from nltk import ngrams
from parlai.agents.image_seq2seq.image_seq2seq import ImageSeq2seqAgent
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.metrics import AverageMetric, SumMetric, GlobalAverageMetric
from parlai.utils.misc import round_sigfigs
class RepetitionUnlikelihoodAgentTrait(object):
    """
    Abstract Trait.

    Applies unliikelihood loss to repetition some proportion of train steps by
    generating, marking repeats, and calculating loss accordingly.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.pred_logsoftmax = torch.nn.LogSoftmax(dim=2)

    @classmethod
    def add_cmdline_args(cls, argparser):
        print(super())
        grp = super().add_cmdline_args(argparser)
        grp.add_argument('--seq-ul-ratio', default=0.5, type=float)
        grp.add_argument('--seq-ul-n', default=4, type=int)
        grp.add_argument('--mask-n', default=100, type=int)
        grp.add_argument('--ctxt-beta', default=0.5, type=float)
        grp.add_argument('--crep-pen', default='crep', type=str)

    def _init_cuda_buffer(self, batchsize, maxlen, force=False):
        pass

    def _count_n_grams(self, token_lst, n):
        n_grams = defaultdict(int)
        for n_gram in NGramIterator(token_lst, n):
            n_grams[n_gram] += 1
        return n_grams

    def compute_loss(self, batch, return_output=False):
        if self.is_training and torch.rand(1).item() >= self.opt['seq_ul_ratio']:
            total_loss, model_output = super().compute_loss(batch, return_output=True)
            if return_output:
                return (total_loss, model_output)
            else:
                return total_loss
        clamp_min = 1e-06 if self.opt['fp16'] else 1e-20
        maxlen = self.label_truncate or 256
        with torch.no_grad():
            beam_pred_scores, _ = self._generate(batch, self.beam_size, maxlen)
        generations = [g[1:] for g, s in beam_pred_scores]
        pred_toks = torch.nn.utils.rnn.pad_sequence(generations, batch_first=True)
        model_output = self.model(*self._model_input(batch), ys=pred_toks)
        logits, preds, _ = model_output
        n = self.opt['seq_ul_n']
        crep_mask = torch.zeros_like(pred_toks).type_as(logits)
        lrep_mask = torch.zeros_like(pred_toks).type_as(logits)
        for i, gen in enumerate(generations):
            gen_i = gen.tolist()
            context_i = batch.text_vec[i].tolist()
            context_n_grams = self._count_n_grams(context_i, n)
            seen_n_grams = defaultdict(int)
            for j, n_gram in enumerate(NGramIterator(gen_i, n)):
                if context_n_grams[n_gram] > 0:
                    crep_mask[i, j:j + n] = 1
            for j, n_gram in enumerate(NGramIterator(gen_i, n)):
                if seen_n_grams[n_gram] > 0:
                    lrep_mask[i, j:j + n] = 1
                seen_n_grams[n_gram] += 1
        lprobs = self.pred_logsoftmax(logits)
        pred_lprobs = lprobs.view(-1, lprobs.size(2)).gather(1, pred_toks.view(-1, 1))
        one_minus_probs = torch.clamp(1.0 - pred_lprobs.exp(), min=clamp_min).view(pred_toks.size(0), pred_toks.size(1))
        mask = (1 - self.opt['ctxt_beta']) * lrep_mask + self.opt['ctxt_beta'] * crep_mask
        ul_loss = -torch.log(one_minus_probs) * mask
        total_loss = div(ul_loss.sum(), mask.sum())
        self.record_local_metric('ul_loss', AverageMetric.many(ul_loss.sum(dim=-1), mask.sum(dim=-1)))
        if not self.is_training:
            _, _ = super().compute_loss(batch, return_output=True)
        if return_output:
            return (total_loss, model_output)
        return total_loss

    def _add_generation_metrics(self, batch, preds):
        self._ngram_metrics(batch, preds)

    def _ngram_metrics(self, batch, preds):
        text_vecs_cpu = batch.text_vec.cpu()
        lrep, crep = (0, 0)
        total_pred_ngs = 0
        n = self.opt['seq_ul_n']
        for i, pred in enumerate(preds):
            pred_token_list = pred.tolist()
            if self.END_IDX in pred_token_list:
                pred_token_list = pred_token_list[:pred_token_list.index(self.END_IDX)]
            if self.START_IDX in pred_token_list:
                pred_token_list = pred_token_list[pred_token_list.index(self.START_IDX):]
            pred_ngs = [ng for ng in ngrams(pred_token_list, n)]
            pred_counter = Counter(pred_ngs)
            total_pred_ngs += len(pred_ngs)
            lrep += len(pred_ngs) - len(pred_counter)
            text_token_list = text_vecs_cpu[i].tolist()
            if self.NULL_IDX in text_token_list:
                text_token_list = text_token_list[:text_token_list.index(self.NULL_IDX)]
            context_counter = Counter([ng for ng in ngrams(text_token_list, n)])
            for ng in pred_counter:
                if ng in context_counter:
                    crep += pred_counter[ng]
        self.global_metrics.add('lrep_%dgrams' % n, GlobalAverageMetric(lrep, total_pred_ngs))
        self.global_metrics.add('crep_%dgrams' % n, GlobalAverageMetric(crep, total_pred_ngs))