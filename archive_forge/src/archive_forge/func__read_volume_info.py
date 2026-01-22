import getpass
import time
import warnings
from collections import OrderedDict
import numpy as np
from ..openers import Opener
def _read_volume_info(fobj):
    """Helper for reading the footer from a surface file."""
    volume_info = OrderedDict()
    head = np.fromfile(fobj, '>i4', 1)
    if not np.array_equal(head, [20]):
        head = np.concatenate([head, np.fromfile(fobj, '>i4', 2)])
        if not np.array_equal(head, [2, 0, 20]):
            warnings.warn('Unknown extension code.')
            return volume_info
    volume_info['head'] = head
    for key in ('valid', 'filename', 'volume', 'voxelsize', 'xras', 'yras', 'zras', 'cras'):
        pair = fobj.readline().decode('utf-8').split('=')
        if pair[0].strip() != key or len(pair) != 2:
            raise OSError('Error parsing volume info.')
        if key in ('valid', 'filename'):
            volume_info[key] = pair[1].strip()
        elif key == 'volume':
            volume_info[key] = np.array(pair[1].split(), int)
        else:
            volume_info[key] = np.array(pair[1].split(), float)
    return volume_info