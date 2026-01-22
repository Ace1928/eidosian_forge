import cupy
def _convolve1d2o(in1, in2, mode):
    assert mode == 'valid'
    out_dim = in1.shape[0] - max(in2.shape) + 1
    dtype = cupy.result_type(in1, in2)
    out = cupy.empty(out_dim, dtype=dtype)
    _convolve1d2o_kernel(in1, in2, *in2.shape, out)
    return out