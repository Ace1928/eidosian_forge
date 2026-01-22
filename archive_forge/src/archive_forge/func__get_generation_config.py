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
def _get_generation_config(is_multilingual: bool, num_languages: int=100, openai_version: Optional[str]=None) -> GenerationConfig:
    """
    Loads the appropriate generation config from HF repo
    """
    if openai_version is not None:
        repo = f'openai/whisper-{openai_version}'
    elif not is_multilingual:
        repo = 'openai/whisper-medium.en'
    elif num_languages < 100:
        repo = 'openai/whisper-large-v2'
    else:
        repo = 'openai/whisper-large-v3'
    gen_cfg = GenerationConfig.from_pretrained(repo)
    if openai_version is None:
        gen_cfg.alignment_heads = None
        warnings.warn('Alignment heads have not been included in the generation config, since they are available only for the original OpenAI checkpoints.If you want to use word-level timestamps with a custom version of Whisper,see https://github.com/openai/whisper/blob/main/notebooks/Multilingual_ASR.ipynbfor the example of how to produce word-level timestamps manually.')
    return gen_cfg