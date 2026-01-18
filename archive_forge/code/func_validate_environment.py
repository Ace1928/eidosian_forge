import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Union
from packaging import version
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name
from ..utils import is_accelerate_available, is_bitsandbytes_available, is_torch_available, logging
def validate_environment(self, *args, **kwargs):
    if not (is_accelerate_available() and is_bitsandbytes_available()):
        raise ImportError('Using `bitsandbytes` 8-bit quantization requires Accelerate: `pip install accelerate` and the latest version of bitsandbytes: `pip install -i https://pypi.org/simple/ bitsandbytes`')
    if kwargs.get('from_tf', False) or kwargs.get('from_flax', False):
        raise ValueError('Converting into 4-bit or 8-bit weights from tf/flax weights is currently not supported, please make sure the weights are in PyTorch format.')
    if not torch.cuda.is_available():
        raise RuntimeError('No GPU found. A GPU is needed for quantization.')
    device_map = kwargs.get('device_map', None)
    if device_map is not None and isinstance(device_map, dict) and (not self.quantization_config.llm_int8_enable_fp32_cpu_offload):
        device_map_without_lm_head = {key: device_map[key] for key in device_map.keys() if key not in self.modules_to_not_convert}
        if 'cpu' in device_map_without_lm_head.values() or 'disk' in device_map_without_lm_head.values():
            raise ValueError('\n                    Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the\n                    quantized model. If you want to dispatch the model on the CPU or the disk while keeping these modules\n                    in 32-bit, you need to set `load_in_8bit_fp32_cpu_offload=True` and pass a custom `device_map` to\n                    `from_pretrained`. Check\n                    https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu\n                    for more details.\n                    ')
    if version.parse(importlib.metadata.version('bitsandbytes')) < version.parse('0.39.0'):
        raise ValueError('You have a version of `bitsandbytes` that is not compatible with 4bit inference and training make sure you have the latest version of `bitsandbytes` installed')