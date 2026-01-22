from dataclasses import dataclass
from functools import partial
from typing import Callable
import torch
import torchaudio
from torchaudio.models import conv_tasnet_base, hdemucs_high
Construct the model and load the pretrained weight.