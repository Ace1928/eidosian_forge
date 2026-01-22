from collections import OrderedDict
import os
import torch
from torch.serialization import default_restore_location
from typing import Any, Dict, List
from parlai.core.agents import create_agent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript
def _validate_fairseq_args(self, args: Dict[str, Any]):
    """
        Validate that fairseq args are compatible with ParlAI.
        """
    norm_key = 'encoder_normalize_before'
    assert args[norm_key] == args[norm_key], 'This asymmetrical transformer is not supported yet!'
    assert not ('layernorm_extra' in args and args['layernorm_extra']), 'Please handle layernorm extra'
    for k in TRANSFORMER_PARAMETER_MAPPING:
        assert args[f'encoder_{k}'] == args[f'decoder_{k}'], 'This asymmetrical transformer is not supported yet!'