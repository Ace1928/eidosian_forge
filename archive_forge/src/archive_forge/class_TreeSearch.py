from abc import ABC, abstractmethod
from typing import TypeVar, List, Dict, Optional, Tuple, Set, Iterable
import math
from operator import attrgetter
import torch
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.opt import Opt
from parlai.utils.distributed import is_distributed, sync_parameters
from parlai.core.torch_agent import TorchAgent, Batch, Output, DictionaryAgent
from parlai.utils.misc import warn_once
import parlai.utils.logging as logging
from parlai.core.metrics import (
from parlai.utils.fp16 import FP16SafeCrossEntropy
from parlai.utils.torch import (
class TreeSearch(object):
    """
    Abstract Tree Search class.

    It keeps information about beam_size concurrent, developing hypotheses. Concrete
    implementations make choices about which token to explore next at each point in the
    tree. Different choices result in different generation algorithms.
    """

    def __init__(self, beam_size, block_ngram=-1, context_block_ngram=-1, padding_token=0, bos_token=1, eos_token=2, min_length=3, device='cpu', length_penalty=0.65):
        """
        Instantiate Beam object.

        :param beam_size:
            number of hypothesis in the beam
        :param block_ngram:
            size of ngrams to block.
        :param context_block_ngram:
            size of context ngrams to block
        :param padding_token:
            padding token ID
        :param bos_token:
            beginning of sentence token ID
        :param eos_token:
            end of sentence token ID
        :param min_length:
            minimum length of the predicted sequence
        :param device:
            What device to use for computations
        """
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.block_ngram = block_ngram
        self.min_length = min_length
        self.eos = eos_token
        self.bos = bos_token
        self.pad = padding_token
        self.context = None
        self.context_block_ngram = context_block_ngram
        self.block_list: Optional[SearchBlocklist] = None
        self.device = device
        self.scores = None
        self.all_scores = [torch.Tensor([0.0] * beam_size).to(self.device)]
        self.bookkeep = []
        self.outputs = [torch.Tensor(self.beam_size).long().fill_(self.bos).to(self.device)]
        self.finished = []
        self.eos_top = False
        self.eos_top_ts = None
        self.n_best_counter = 0
        self.partial_hyps = [[self.bos] for i in range(beam_size)]

    def set_context(self: TSType, context: torch.LongTensor) -> TSType:
        """
        Set the internal context representation and return self.

        :param context:
            a LongTensor representing the input context; used for context
            ngram blocking, if supplied
        """
        self.context = context.tolist()
        return self

    def set_block_list(self: TSType, block_list: Optional[SearchBlocklist]) -> TSType:
        self.block_list = block_list
        return self

    def get_output_from_current_step(self):
        """
        Get the outputput at the current step.
        """
        return self.outputs[-1]

    def get_backtrack_from_current_step(self):
        """
        Get the backtrack at the current step.
        """
        return self.bookkeep[-1]

    @abstractmethod
    def select_paths(self, logprobs, prior_scores, current_length):
        """
        Select the next vocabulary item in these beams.

        :param logprobs:
            a (beamsize x vocab) tensor of log probabilities. If this is the first
            turn in the dialogue, it will be a (1 x vocab) tensor.
        :param prior_scores:
            a (beamsize) tensor of weights with the cumulative running
            log-probability of each beam. If the first turn, it will be a (1) tensor.
        :param current_length:
            the current length in tokens
        :return:
            a (hypothesis_ids, token_id, scores) tuple, where:

            - hypothesis_ids is a LongTensor of hypotheses we're extending. May have
              repeats, but should always be (beamsize) long.
            - token_ids is a (beamsize) LongTensor of next-token choices for
              each of the hypotheses.
            - scores is a (beamsize) Tensor with the updated cumulative log-probs
              of each beam.
        """
        pass

    def _block_ngrams(self, ngram_size: int, logprobs: torch.Tensor, source: torch.LongTensor=None):
        """
        Hard block ngrams from the logprobs, based on the source.

        :param ngram_size:
            The length of ngrams to block. Must be > 0.
        :param logprobs:
            Float or HalfTensor, representing the log-probabilities. This is
            modified in place.
        :param source:
            Source text to grab ngrams from. If None, it uses the current
            hypothesis (i.e. self-blocking).
        """
        for beam_id, hyp in enumerate(self.partial_hyps):
            if len(hyp) < ngram_size - 1:
                continue
            source_ = hyp if source is None else source
            ngrams = self._find_ngrams(source_, ngram_size)
            prefix = hyp[-(ngram_size - 1):]
            for ngram in ngrams:
                if ngram_size == 1 or prefix == list(ngram[:-1]):
                    logprobs[beam_id][ngram[-1]] = neginf(logprobs.dtype)
        return logprobs

    def _block_block_list(self, logprobs: torch.Tensor) -> torch.Tensor:
        if self.block_list is None:
            return logprobs
        for beam_id, hyp in enumerate(self.partial_hyps):
            for ngram_size, bad_ngrams in self.block_list.items():
                prefix = hyp[-(ngram_size - 1):]
                for ngram in bad_ngrams:
                    if ngram_size == 1 or prefix == list(ngram[:-1]):
                        logprobs[beam_id][ngram[-1]] = neginf(logprobs.dtype)
        return logprobs

    def advance(self, logprobs):
        """
        Advance the beam one step.
        """
        current_length = len(self.all_scores) - 1
        if current_length < self.min_length:
            for hyp_id in range(logprobs.size(0)):
                logprobs[hyp_id][self.eos] = neginf(logprobs.dtype)
        if self.scores is None:
            self.scores = torch.zeros(1).type_as(logprobs).to(logprobs.device)
        for hyp_id, token in enumerate(self.outputs[-1]):
            if token == self.eos:
                self.scores[hyp_id] = neginf(self.scores.dtype)
        if self.block_ngram > 0:
            logprobs = self._block_ngrams(self.block_ngram, logprobs, None)
        logprobs = self._block_block_list(logprobs)
        if self.context_block_ngram > 0:
            if self.context is None:
                raise ValueError('Must use TreeSearch.set_context to use context blocking.')
            logprobs = self._block_ngrams(self.context_block_ngram, logprobs, self.context)
        hyp_ids, tok_ids, self.scores = self.select_paths(logprobs, self.scores, current_length)
        self.all_scores.append(self.scores.clone())
        self.outputs.append(tok_ids)
        self.bookkeep.append(hyp_ids)
        self.partial_hyps = [self.partial_hyps[hyp_ids[i]] + [tok_ids[i].item()] for i in range(self.beam_size)]
        for hypid in range(self.beam_size):
            if self.outputs[-1][hypid] == self.eos:
                if self.scores[hypid] <= neginf(self.scores.dtype):
                    continue
                eostail = _HypothesisTail(timestep=len(self.outputs) - 1, hypid=hypid, score=self.all_scores[-1][hypid], tokenid=self.eos)
                self.finished.append(eostail)
                self.n_best_counter += 1
        if self.outputs[-1][0] == self.eos:
            self.eos_top = True
            if self.eos_top_ts is None:
                self.eos_top_ts = len(self.outputs) - 1

    def is_done(self):
        """
        Return whether beam search is complete.
        """
        return self.eos_top and self.n_best_counter >= self.beam_size

    def _find_ngrams(self, input_list, n):
        """
        Find ngrams of size n in input list.
        """
        return list(zip(*[input_list[i:] for i in range(n)]))

    def _get_hyp_from_finished(self, hypothesis_tail):
        """
        Extract hypothesis ending with EOS at timestep with hyp_id.

        :param timestep:
            timestep with range up to len(self.outputs) - 1

        :param hyp_id:
            id with range up to beam_size - 1

        :return:
            hypothesis sequence
        """
        hyp_idx = []
        endback = hypothesis_tail.hypid
        for i in range(hypothesis_tail.timestep, -1, -1):
            hyp_idx.append(_HypothesisTail(timestep=i, hypid=endback, score=self.all_scores[i][endback], tokenid=self.outputs[i][endback]))
            endback = self.bookkeep[i - 1][endback]
        return hyp_idx

    def _get_pretty_hypothesis(self, list_of_hypotails):
        """
        Return hypothesis as a tensor of token ids.
        """
        return torch.stack([ht.tokenid for ht in reversed(list_of_hypotails)])

    def get_rescored_finished(self, n_best=None):
        """
        Return finished hypotheses according to adjusted scores.

        Score adjustment is done according to the Google NMT paper, which
        penalizes long utterances.

        :param n_best:
            number of finalized hypotheses to return

        :return:
            list of (tokens, score) pairs, in sorted order, where:
              - tokens is a tensor of token ids
              - score is the adjusted log probability of the entire utterance
        """
        if not self.finished:
            self.outputs[-1][0] = self.eos
            self.finished.append(_HypothesisTail(timestep=len(self.outputs) - 1, hypid=0, score=self.all_scores[-1][0], tokenid=self.outputs[-1][0]))
        rescored_finished = []
        for finished_item in self.finished:
            current_length = finished_item.timestep + 1
            length_penalty = math.pow((1 + current_length) / 6, self.length_penalty)
            rescored_finished.append(_HypothesisTail(timestep=finished_item.timestep, hypid=finished_item.hypid, score=finished_item.score / length_penalty, tokenid=finished_item.tokenid))
        srted = sorted(rescored_finished, key=attrgetter('score'), reverse=True)
        if n_best is not None:
            srted = srted[:n_best]
        n_best_list = [(self._get_pretty_hypothesis(self._get_hyp_from_finished(hyp)), hyp.score) for hyp in srted]
        assert len(n_best_list) >= 1, f'TreeSearch returned {len(n_best_list)} candidates, must be >= 1'
        for pred, score in n_best_list:
            assert (pred == self.eos).sum() == 1, f'TreeSearch returned a finalized hypo with multiple end tokens             with score {score.item():.2f}'
        return n_best_list