from typing import Dict, Any, Union, List, Tuple, Optional
from abc import ABC, abstractmethod
import random
import os
import torch
import parlai.utils.logging as logging
from torch import optim
from parlai.core.opt import Opt
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.nn.lr_scheduler import ParlAILRScheduler
from parlai.core.message import Message
from parlai.utils.distributed import is_distributed
from parlai.utils.misc import AttrDict, warn_once
from parlai.utils.fp16 import (
from parlai.core.metrics import (
from parlai.utils.distributed import is_primary_worker
from parlai.utils.torch import argsort, compute_grad_norm, padded_tensor, atomic_save
def _set_label_vec(self, obs, add_start, add_end, truncate):
    """
        Set the 'labels_vec' field in the observation.

        Useful to override to change vectorization behavior
        """
    if 'labels' in obs:
        label_type = 'labels'
    elif 'eval_labels' in obs:
        label_type = 'eval_labels'
    else:
        label_type = None
    if label_type is None:
        return
    elif label_type + '_vec' in obs:
        truncated_vec = self._check_truncate(obs[label_type + '_vec'], truncate)
        obs.force_set(label_type + '_vec', torch.LongTensor(truncated_vec))
    else:
        lbls = obs[label_type]
        label = lbls[0] if len(lbls) == 1 else self.random.choice(lbls)
        vec_label = self._vectorize_text(label, add_start, add_end, truncate, False)
        obs[label_type + '_vec'] = vec_label
        obs[label_type + '_choice'] = label
    return obs