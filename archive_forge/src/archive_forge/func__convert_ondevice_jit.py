import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quant_type import QuantType
from torch.jit._recursive import wrap_cpp_module
def _convert_ondevice_jit(model, method_name, inplace=False, debug=False, quant_type=QuantType.STATIC):
    _check_is_script_module(model)
    assert quant_type == QuantType.DYNAMIC, 'This API, while should work for static quant, is only tested for dynamic quant.'
    assert not method_name.startswith('observe_'), 'Pass in valid method to be quantized, e.g. forward'
    observe_method_name = 'observe_' + method_name
    quantize_method_name = 'quantize_' + method_name
    model_c = model._c
    model_c = torch._C._jit_pass_insert_quant_dequant_for_ondevice_ptq(model._c, observe_method_name, inplace, debug, QuantType.DYNAMIC)
    model_c = torch._C._jit_pass_quant_finalize_for_ondevice_ptq(model_c, QuantType.DYNAMIC, quantize_method_name)
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    return model