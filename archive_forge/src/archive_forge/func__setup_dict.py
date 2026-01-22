from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.utils.misc import round_sigfigs
from .modules import TransresnetModel
from parlai.tasks.personality_captions.build import build
import os
import random
import json
import numpy as np
import torch
import tqdm
def _setup_dict(self):
    """
        Set up the dictionary.

        The pretrained model used a separate dictionary from the standard ParlAI one.
        """
    self.dict = DictionaryAgent(self.opt)
    if self.opt.get('pretrained', False):
        new_tok2ind = {}
        new_ind2tok = {}
        for key in self.dict.tok2ind:
            val = self.dict.tok2ind[key]
            if val - 4 >= 0:
                new_tok2ind[key] = val - 4
                new_ind2tok[val - 4] = key
        self.dict.null_token = '<PAD>'
        self.dict.unk_token = '<UNK>'
        self.dict.tok2ind = new_tok2ind
        self.dict.ind2tok = new_ind2tok