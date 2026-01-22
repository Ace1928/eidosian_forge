from parlai.core.agents import Agent
from parlai.utils.torch import padded_3d
from parlai.core.torch_classifier_agent import TorchClassifierAgent
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.utils.misc import recursive_getattr
from parlai.utils.logging import logging
from .modules import (
import torch
class TransformerRankerAgent(TorchRankerAgent):
    """
    Transformer Ranker Agent.

    Implementation of a TorchRankerAgent, where the model is a Transformer
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        super(TransformerRankerAgent, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Transformer Arguments')
        add_common_cmdline_args(agent)
        agent.add_argument('--use-memories', type='bool', default=False, help='use memories: must implement the function `_vectorize_memories` to use this')
        agent.add_argument('--wrap-memory-encoder', type='bool', default=False, help='wrap memory encoder with MLP')
        agent.add_argument('--memory-attention', type=str, default='sqrt', choices=['cosine', 'dot', 'sqrt'], help='similarity for basic attention mechanism when using transformer to encode memories')
        agent.add_argument('--normalize-sent-emb', type='bool', default=False)
        agent.add_argument('--share-encoders', type='bool', default=True)
        argparser.add_argument('--share-word-embeddings', type='bool', default=True, help='Share word embeddings table for candidate and contextin the memory network')
        agent.add_argument('--learn-embeddings', type='bool', default=True, help='learn embeddings')
        agent.add_argument('--data-parallel', type='bool', default=False, help='use model in data parallel, requires multiple gpus')
        agent.add_argument('--reduction-type', type=str, default='mean', choices=['first', 'max', 'mean'], help='Type of reduction at the end of transformer')
        argparser.set_defaults(learningrate=0.0001, optimizer='adamax', truncate=1024)
        cls.dictionary_class().add_cmdline_args(argparser)
        return agent

    def _score(self, output, cands):
        if cands.dim() == 2:
            return torch.matmul(output, cands.t())
        elif cands.dim() == 3:
            return torch.bmm(output.unsqueeze(1), cands.transpose(1, 2)).squeeze(1)
        else:
            raise RuntimeError('Unexpected candidate dimensions {}'.format(cands.dim()))

    def build_model(self, states=None):
        """
        Build and return model.
        """
        model = TransformerMemNetModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(model.embeddings.weight, self.opt['embedding_type'])
        return model

    def batchify(self, obs_batch, sort=False):
        """
        Override so that we can add memories to the Batch object.
        """
        batch = super().batchify(obs_batch, sort)
        if self.opt['use_memories']:
            valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if self.is_valid(ex)]
            valid_inds, exs = zip(*valid_obs)
            mems = None
            if any(('memory_vecs' in ex for ex in exs)):
                mems = [ex.get('memory_vecs', None) for ex in exs]
            batch.memory_vecs = mems
        return batch

    def _vectorize_memories(self, obs):
        raise NotImplementedError('Abstract class: user must implement this function to use memories')

    def vectorize(self, *args, **kwargs):
        """
        Override to include vectorization of memories.
        """
        kwargs['add_start'] = False
        kwargs['add_end'] = False
        obs = super().vectorize(*args, **kwargs)
        if self.opt['use_memories']:
            obs = self._vectorize_memories(obs)
        return obs

    def encode_candidates(self, padded_cands):
        """
        Encode candidates.
        """
        _, cands = self.model(xs=None, mems=None, cands=padded_cands)
        return cands

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        """
        Score candidates.
        """
        if self.opt['use_memories'] and batch.memory_vecs is not None and sum((len(m) for m in batch.memory_vecs)):
            mems = padded_3d(batch.memory_vecs, use_cuda=self.use_cuda, pad_idx=self.NULL_IDX)
        else:
            mems = None
        if cand_encs is not None:
            cand_vecs = None
        context_h, cands_h = self.model(xs=batch.text_vec, mems=mems, cands=cand_vecs)
        if cand_encs is not None:
            cands_h = cand_encs
        scores = self._score(context_h, cands_h)
        return scores