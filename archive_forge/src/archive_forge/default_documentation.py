from pathlib import Path
import torch
from ...utils import is_npu_available, is_xpu_available
from .config_args import ClusterConfig, default_json_config_file
from .config_utils import SubcommandHelpFormatter

    Creates and saves a basic cluster config to be used on a local machine with potentially multiple GPUs. Will also
    set CPU if it is a CPU-only machine.

    Args:
        mixed_precision (`str`, *optional*, defaults to "no"):
            Mixed Precision to use. Should be one of "no", "fp16", or "bf16"
        save_location (`str`, *optional*, defaults to `default_json_config_file`):
            Optional custom save location. Should be passed to `--config_file` when using `accelerate launch`. Default
            location is inside the huggingface cache folder (`~/.cache/huggingface`) but can be overriden by setting
            the `HF_HOME` environmental variable, followed by `accelerate/default_config.yaml`.
        use_xpu (`bool`, *optional*, defaults to `False`):
            Whether to use XPU if available.
    