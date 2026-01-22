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
class RewardUnlikelihoodAgentTrait(object):
    """
    Abstract Trait.

    Applies unlikelihood loss based on the presence of negative rewards in the task.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        grp = super(RewardUnlikelihoodAgentTrait, cls).add_cmdline_args(argparser)
        grp.add_argument('--alpha', default=1.0, type=float)

    def batchify(self, obs_batch, **kwargs):
        batch = super().batchify(obs_batch, **kwargs)
        rewards = torch.FloatTensor([float(o.get('reward', 0)) for o in batch.observations]).to(batch.text_vec.device)
        batch['rewards'] = rewards
        return batch

    def _dummy_batch(self, batchsize, maxlen):
        batch = super()._dummy_batch(batchsize, maxlen)
        batch['rewards'] = torch.ones(batchsize, dtype=torch.long).cuda()
        return batch

    def compute_loss(self, batch, return_output=False):
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, *_ = model_output
        scores = F.log_softmax(scores, dim=-1)
        scores_view = scores.view(-1, scores.size(-1))
        targets = batch.label_vec
        targets_view = targets.view(-1)
        notnull = targets.ne(self.NULL_IDX)
        if self.is_training:
            mle_notnull = notnull & (batch.rewards >= 0).unsqueeze(1).expand_as(notnull)
        else:
            mle_notnull = notnull
        mle_loss = (F.nll_loss(scores_view, targets_view, ignore_index=self.NULL_IDX, reduction='none') * mle_notnull.view(-1).float()).sum()
        mle_target_tokens = mle_notnull.long().sum().item()
        correct = ((targets == preds) * mle_notnull).sum().item()
        self.record_local_metric('correct_tokens', SumMetric(correct))
        self.record_local_metric('nll_loss', SumMetric(mle_loss.item()))
        self.record_local_metric('num_tokens', SumMetric(mle_target_tokens))
        if mle_target_tokens > 0:
            mle_loss /= mle_target_tokens
        if not self.is_training:
            if return_output:
                return (mle_loss, model_output)
            else:
                return mle_loss
        ul_notnull = notnull & (batch.rewards < 0).unsqueeze(1).expand_as(notnull)
        ul_target_tokens = ul_notnull.long().sum().item()
        range_ = torch.arange(targets_view.size(0)).to(batch.label_vec.device)
        ul_scores = scores_view[range_, targets_view]
        clamp_min = 1e-06 if self.opt['fp16'] else 1e-20
        ul_loss = (-torch.log(torch.clamp(1.0 - ul_scores.exp(), min=clamp_min)) * ul_notnull.view(-1).float()).sum()
        self.record_local_metric('ul_loss', AverageMetric.many(ul_loss.sum(dim=-1), ul_target_tokens))
        if ul_target_tokens > 0:
            ul_loss /= ul_target_tokens
        loss = mle_loss + self.opt['alpha'] * ul_loss
        if return_output:
            return (loss, model_output)
        else:
            return loss