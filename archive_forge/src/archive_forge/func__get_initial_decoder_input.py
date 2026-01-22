import os
import torch
from typing import Optional, Dict, Any
from parlai.agents.bart.convert_fairseq_to_parlai import ConversionScript
from parlai.agents.bart.modules import BartModel
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.agents import compare_init_model_opts
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import Batch, History, TorchAgent
from parlai.utils.typing import TShared
from parlai.zoo.bart.build import download, CONVERSION_ARGS, BART_ARGS
def _get_initial_decoder_input(self, bsz: int, beam_size: int, dev: torch.device) -> torch.LongTensor:
    """
        Override to seed decoder with EOS token.

        See docstring for `BartAgent._generate` for more details.
        """
    return torch.LongTensor([self.END_IDX]).expand(bsz * beam_size, 1).to(dev)