from typing import Callable, Dict, List, Optional, Tuple
import torch
from torchaudio.models import RNNT
from torchaudio.prototype.models.rnnt import TrieNode
def _set_hypo_trie(hypo: Hypothesis, trie: TrieNode) -> None:
    hypo[4] = trie