import numpy
from cupy import _core
from cupyx.jit import _interface
from cupyx.jit import _cuda_types
@staticmethod
def _get_body(return_type, call):
    if isinstance(return_type, _cuda_types.Scalar):
        dtypes = [return_type.dtype]
        code = f'out0 = {call};'
    elif isinstance(return_type, _cuda_types.Tuple):
        dtypes = []
        code = f'auto out = {call};\n'
        for i, t in enumerate(return_type.types):
            if not isinstance(t, _cuda_types.Scalar):
                raise TypeError(f'Invalid return type: {return_type}')
            dtypes.append(t.dtype)
            code += f'out{i} = thrust::get<{i}>(out);\n'
    else:
        raise TypeError(f'Invalid return type: {return_type}')
    out_params = [f'{dtype} out{i}' for i, dtype in enumerate(dtypes)]
    return (', '.join(out_params), code)