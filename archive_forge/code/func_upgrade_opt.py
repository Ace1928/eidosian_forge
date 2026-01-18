import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.torch_agent import TorchAgent, Output
from parlai.utils.misc import round_sigfigs, warn_once
from parlai.utils.torch import padded_tensor
from parlai.agents.transformer.transformer import TransformerRankerAgent
from .feedback_classifier.feedback_classifier import FeedbackClassifierRegex
from .modules import SelfFeedingModel
@classmethod
def upgrade_opt(cls, opt_on_disk):
    """
        Upgrade opts from older model files.
        """
    super(SelfFeedingAgent, cls).upgrade_opt(opt_on_disk)
    if 'add_double_person_tokens' not in opt_on_disk:
        warn_once('Old model: overriding `add_double_person_tokens` to True.')
        opt_on_disk['add_double_person_tokens'] = True
    return opt_on_disk