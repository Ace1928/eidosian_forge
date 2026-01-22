import argparse
import os
from pathlib import Path
import torch
from bark.generation import _load_model as _bark_load_model
from huggingface_hub import hf_hub_download
from transformers import EncodecConfig, EncodecModel, set_seed
from transformers.models.bark.configuration_bark import (
from transformers.models.bark.generation_configuration_bark import (
from transformers.models.bark.modeling_bark import BarkCoarseModel, BarkFineModel, BarkModel, BarkSemanticModel
from transformers.utils import logging
def _get_ckpt_path(model_type, use_small=False):
    key = model_type
    if use_small:
        key += '_small'
    return os.path.join(CACHE_DIR, REMOTE_MODEL_PATHS[key]['file_name'])