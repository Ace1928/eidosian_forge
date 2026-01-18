import os
import warnings
from typing import Optional
import torch
from huggingface_hub import file_exists, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors.torch import load_file as safe_load_file
from .other import (
from .peft_types import PeftType

    A helper method to load the PEFT weights from the HuggingFace Hub or locally

    Args:
        model_id (`str`):
            The local path to the adapter weights or the name of the adapter to load from the HuggingFace Hub.
        device (`str`):
            The device to load the weights onto.
        hf_hub_download_kwargs (`dict`):
            Additional arguments to pass to the `hf_hub_download` method when loading from the HuggingFace Hub.
    