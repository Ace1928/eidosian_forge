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
def _copy_embeddings(self, weight, emb_type, log=True):
    """
        Copy embeddings from the pretrained embeddings to the lookuptable.

        :param weight:
            weights of lookup table (nn.Embedding/nn.EmbeddingBag)

        :param emb_type:
            pretrained embedding type
        """
    if self.opt['embedding_type'] == 'random' or not self._should_initialize_optimizer():
        return
    embs, name = self._get_embtype(emb_type)
    cnt = 0
    for w, i in self.dict.tok2ind.items():
        if w in embs.stoi:
            vec = self._project_vec(embs.vectors[embs.stoi[w]], weight.size(1))
            weight.data[i] = vec
            cnt += 1
    if log:
        logging.info(f'Initialized embeddings for {cnt} tokens ({cnt / len(self.dict):.1%}) from {name}.')