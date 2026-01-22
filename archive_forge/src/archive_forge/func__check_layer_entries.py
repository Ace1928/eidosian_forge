import argparse
import json
import os
import socket
import time
import warnings
from pathlib import Path
from typing import Dict, List, Union
from zipfile import ZipFile
import numpy as np
import torch
from huggingface_hub.hf_api import list_models
from torch import nn
from tqdm import tqdm
from transformers import MarianConfig, MarianMTModel, MarianTokenizer
def _check_layer_entries(self):
    self.encoder_l1 = self.sub_keys('encoder_l1')
    self.decoder_l1 = self.sub_keys('decoder_l1')
    self.decoder_l2 = self.sub_keys('decoder_l2')
    if len(self.encoder_l1) != 16:
        warnings.warn(f'Expected 16 keys for each encoder layer, got {len(self.encoder_l1)}')
    if len(self.decoder_l1) != 26:
        warnings.warn(f'Expected 26 keys for each decoder layer, got {len(self.decoder_l1)}')
    if len(self.decoder_l2) != 26:
        warnings.warn(f'Expected 26 keys for each decoder layer, got {len(self.decoder_l1)}')