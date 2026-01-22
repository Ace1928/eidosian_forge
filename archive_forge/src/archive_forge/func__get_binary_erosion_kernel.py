import operator
import warnings
import numpy
import cupy
from cupy import _core
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
@cupy.memoize(for_each_device=True)
def _get_binary_erosion_kernel(w_shape, int_type, offsets, center_is_true, border_value, invert, masked, all_weights_nonzero):
    if invert:
        border_value = int(not border_value)
        true_val = 0
        false_val = 1
    else:
        true_val = 1
        false_val = 0
    if masked:
        pre = '\n            bool mv = (bool)mask[i];\n            bool _in = (bool)x[i];\n            if (!mv) {{\n                y = cast<Y>(_in);\n                return;\n            }} else if ({center_is_true} && _in == {false_val}) {{\n                y = cast<Y>(_in);\n                return;\n            }}'.format(center_is_true=int(center_is_true), false_val=false_val)
    else:
        pre = '\n            bool _in = (bool)x[i];\n            if ({center_is_true} && _in == {false_val}) {{\n                y = cast<Y>(_in);\n                return;\n            }}'.format(center_is_true=int(center_is_true), false_val=false_val)
    pre = pre + '\n            y = cast<Y>({true_val});'.format(true_val=true_val)
    found = '\n        if ({{cond}}) {{{{\n            if (!{border_value}) {{{{\n                y = cast<Y>({false_val});\n                return;\n            }}}}\n        }}}} else {{{{\n            bool nn = {{value}} ? {true_val} : {false_val};\n            if (!nn) {{{{\n                y = cast<Y>({false_val});\n                return;\n            }}}}\n        }}}}'.format(true_val=int(true_val), false_val=int(false_val), border_value=int(border_value))
    name = 'binary_erosion'
    if false_val:
        name += '_invert'
    return _filters_core._generate_nd_kernel(name, pre, found, '', 'constant', w_shape, int_type, offsets, 0, ctype='Y', has_weights=True, has_structure=False, has_mask=masked, binary_morphology=True, all_weights_nonzero=all_weights_nonzero)