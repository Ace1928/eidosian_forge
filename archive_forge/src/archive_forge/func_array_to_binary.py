from collections import defaultdict
import numpy as np
def array_to_binary(ar, obj=None, force_contiguous=True):
    if ar is None:
        return None
    if ar.dtype.kind not in ['u', 'i', 'f']:
        raise ValueError('unsupported dtype: %s' % ar.dtype)
    if ar.dtype == np.float64:
        ar = ar.astype(np.float32)
    if ar.dtype == np.int64:
        ar = ar.astype(np.int32)
    if force_contiguous and (not ar.flags['C_CONTIGUOUS']):
        ar = np.ascontiguousarray(ar)
    return {'value': memoryview(ar), 'dtype': str(ar.dtype), 'length': ar.shape[0], 'size': 1 if len(ar.shape) == 1 else ar.shape[1]}