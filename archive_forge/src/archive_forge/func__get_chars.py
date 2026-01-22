import logging
import os
import torch
from torchaudio._internal import download_url_to_file, module_utils as _mod_utils
def _get_chars():
    return ('_', '-', '!', "'", '(', ')', ',', '.', ':', ';', '?', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')