import numpy as np
def _byte_order_str(dtype):
    """ Normalize byteorder to '<' or '>' """
    swapped = np.dtype(int).newbyteorder('S')
    native = swapped.newbyteorder('S')
    byteorder = dtype.byteorder
    if byteorder == '=':
        return native.byteorder
    if byteorder == 'S':
        return swapped.byteorder
    elif byteorder == '|':
        return ''
    else:
        return byteorder