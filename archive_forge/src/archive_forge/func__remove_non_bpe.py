from parlai.core.opt import Opt
from parlai.core.build_data import modelzoo_path
from parlai.utils.bpe import bpe_factory, BPEHelper
from .agents import Agent
from .build_data import make_dir
from collections import defaultdict
import codecs
import copy
import numpy as np
import os
import json
import re
import parlai.utils.logging as logging
from typing import List
def _remove_non_bpe(self):
    """
        Set the dictionary vocab to the bpe vocab, merging counts.
        """
    to_remove = []
    to_add = []
    for token, freq in self.freq.items():
        tokens = self.bpe_tokenize(token)
        if len(tokens) != 1:
            for t in tokens:
                to_add.append((t, freq))
            to_remove.append(token)
    for token in to_remove:
        del self.freq[token]
        idx = self.tok2ind.pop(token)
        del self.ind2tok[idx]
    for token, freq in to_add:
        self.add_token(token)
        self.freq[token] += freq