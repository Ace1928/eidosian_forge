from parlai.core.agents import Agent
from parlai.utils.torch import padded_3d
from parlai.core.torch_classifier_agent import TorchClassifierAgent
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.utils.misc import recursive_getattr
from parlai.utils.logging import logging
from .modules import (
import torch
class TransformerClassifierAgent(TorchClassifierAgent):
    """
    Classifier based on Transformer.
    """

    @staticmethod
    def add_cmdline_args(parser):
        TransformerRankerAgent.add_cmdline_args(parser)
        TorchClassifierAgent.add_cmdline_args(parser)
        parser.add_argument('--load-from-pretrained-ranker', type='bool', default=False, help='load model from base transformer ranking model (used for pretraining)')
        parser.set_defaults(reduction_type='first')

    def build_model(self):
        num_classes = len(self.class_list)
        self.base_model = TransformerMemNetModel(self.opt, self.dict)
        return TransformerLinearWrapper(self.base_model.context_encoder, num_classes)

    def vectorize(self, *args, **kwargs):
        """
        Add the start and end token to the text.
        """
        kwargs['add_start'] = True
        kwargs['add_end'] = True
        obs = super().vectorize(*args, **kwargs)
        return obs

    def _set_text_vec(self, *args, **kwargs):
        """
        Add the start and end token to the text.
        """
        obs = super()._set_text_vec(*args, **kwargs)
        if 'text_vec' in obs and 'added_start_end' not in obs:
            obs.force_set('text_vec', self._add_start_end_tokens(obs['text_vec'], True, True))
            obs['added_start_end'] = True
        if obs.get('text_vec') is not None:
            truncated_vec = self._check_truncate(obs['text_vec'], self.text_truncate, True)
            obs.force_set('text_vec', torch.LongTensor(truncated_vec))
        return obs

    def score(self, batch):
        return self.model(batch.text_vec)

    def load_state_dict(self, state_dict):
        """
        Load the state dict into model.

        This is easily overridable to facilitate transfer of state dicts.
        """
        if self.is_finetune and self.opt['load_from_pretrained_ranker']:
            self.base_model.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_state_dict(state_dict)