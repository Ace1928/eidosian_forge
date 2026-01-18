import argparse
import pathlib
from pathlib import Path
from tempfile import TemporaryDirectory
import esm as esm_module
import torch
from esm.esmfold.v1.misc import batch_encode_sequences as esmfold_encode_sequences
from esm.esmfold.v1.pretrained import esmfold_v1
from transformers.models.esm.configuration_esm import EsmConfig, EsmFoldConfig
from transformers.models.esm.modeling_esm import (
from transformers.models.esm.modeling_esmfold import EsmForProteinFolding
from transformers.models.esm.tokenization_esm import EsmTokenizer
from transformers.utils import logging
def transfer_and_check_weights(original_module, our_module):
    status = our_module.load_state_dict(original_module.state_dict())
    if status.missing_keys:
        raise ValueError(f'Missing keys: {status.missing_keys}')
    if status.unexpected_keys:
        raise ValueError(f'Unexpected keys: {status.unexpected_keys}')