from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SequenceData
from vllm.utils import in_wsl, is_neuron
Tensors for sampling.