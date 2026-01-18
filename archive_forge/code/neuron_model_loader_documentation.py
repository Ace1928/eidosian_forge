from typing import Type
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.config import ModelConfig, DeviceConfig
from vllm.model_executor.models import ModelRegistry
Utilities for selecting and loading models.