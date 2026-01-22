import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quant_type import QuantType
from torch.jit._recursive import wrap_cpp_module
def _convert_ondevice_dynamic_jit(model, method_name, inplace=False, debug=False):
    return _convert_ondevice_jit(model, method_name, inplace, debug, quant_type=QuantType.DYNAMIC)