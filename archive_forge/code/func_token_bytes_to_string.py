import argparse
import io
import json
import os
import tempfile
import urllib
import warnings
from typing import Any, Optional, Tuple
import torch
from huggingface_hub.utils import insecure_hashlib
from torch import nn
from tqdm import tqdm
from transformers import (
from transformers.models.whisper.tokenization_whisper import LANGUAGES, bytes_to_unicode
from transformers.utils.import_utils import _is_package_available
def token_bytes_to_string(b):
    return ''.join([byte_encoder[ord(char)] for char in b.decode('latin-1')])