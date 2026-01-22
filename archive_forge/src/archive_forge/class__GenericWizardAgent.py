from itertools import chain
from functools import lru_cache
import torch as th
import numpy as np
from parlai.utils.torch import padded_tensor
from parlai.utils.misc import round_sigfigs
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from .modules import EndToEndModel
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE
class _GenericWizardAgent(TransformerGeneratorAgent):

    @classmethod
    def add_cmdline_args(cls, argparser):
        argparser.set_defaults(**DEFAULT_OPTS)
        super(_GenericWizardAgent, cls).add_cmdline_args(argparser)

    def batchify(self, obs_batch):
        batch = super().batchify(obs_batch)
        reordered_observations = [obs_batch[i] for i in batch.valid_indices]
        checked_sentences = []
        for obs in reordered_observations:
            checked_sentence = '{} {} {}'.format(obs.get('title', ''), TOKEN_KNOWLEDGE, obs.get('checked_sentence', ''))
            checked_sentences.append(checked_sentence)
        batch['checked_sentence'] = checked_sentences
        return batch