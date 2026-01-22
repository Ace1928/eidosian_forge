import getpass
import time
import warnings
from collections import OrderedDict
import numpy as np
from ..openers import Opener
def _serialize_volume_info(volume_info):
    """Helper for serializing the volume info."""
    keys = ['head', 'valid', 'filename', 'volume', 'voxelsize', 'xras', 'yras', 'zras', 'cras']
    diff = set(volume_info.keys()).difference(keys)
    if len(diff) > 0:
        raise ValueError(f'Invalid volume info: {diff.pop()}.')
    strings = list()
    for key in keys:
        if key == 'head':
            if not (np.array_equal(volume_info[key], [20]) or np.array_equal(volume_info[key], [2, 0, 20])):
                warnings.warn('Unknown extension code.')
            strings.append(np.array(volume_info[key], dtype='>i4').tobytes())
        elif key in ('valid', 'filename'):
            val = volume_info[key]
            strings.append(f'{key} = {val}\n'.encode())
        elif key == 'volume':
            val = volume_info[key]
            strings.append(f'{key} = {val[0]} {val[1]} {val[2]}\n'.encode())
        else:
            val = volume_info[key]
            strings.append(f'{key:6s} = {val[0]:.10g} {val[1]:.10g} {val[2]:.10g}\n'.encode())
    return b''.join(strings)