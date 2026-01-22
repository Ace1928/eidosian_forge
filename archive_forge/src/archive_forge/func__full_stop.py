import torch
import unicodedata
from collections import Counter
from parlai.core.build_data import modelzoo_path
def _full_stop(w):
    return w in {'.', '?', '!'}