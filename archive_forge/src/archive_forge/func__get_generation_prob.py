from typing import Callable, Dict, List, Optional, Tuple
import torch
from torchaudio.models import RNNT
from torchaudio.prototype.models.rnnt import TrieNode
def _get_generation_prob(self, trie):
    if len(trie[0].keys()) == 0:
        return True
    else:
        return False