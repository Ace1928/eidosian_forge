import fnmatch
import io
import re
import tarfile
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from ray.data.block import BlockAccessor
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
def _default_decoder(sample: Dict[str, Any], format: Optional[Union[bool, str]]=True):
    """A default decoder for webdataset.

    This handles common file extensions: .txt, .cls, .cls2,
        .jpg, .png, .json, .npy, .mp, .pt, .pth, .pickle, .pkl.
    These are the most common extensions used in webdataset.
    For other extensions, users can provide their own decoder.

    Args:
        sample: sample, modified in place
    """
    sample = dict(sample)
    for key, value in sample.items():
        extension = key.split('.')[-1]
        if key.startswith('__'):
            continue
        elif extension in ['txt', 'text']:
            sample[key] = value.decode('utf-8')
        elif extension in ['cls', 'cls2']:
            sample[key] = int(value.decode('utf-8'))
        elif extension in ['jpg', 'png', 'ppm', 'pgm', 'pbm', 'pnm']:
            import numpy as np
            import PIL.Image
            if format == 'PIL':
                sample[key] = PIL.Image.open(io.BytesIO(value))
            else:
                sample[key] = np.asarray(PIL.Image.open(io.BytesIO(value)))
        elif extension == 'json':
            import json
            sample[key] = json.loads(value)
        elif extension == 'npy':
            import numpy as np
            sample[key] = np.load(io.BytesIO(value))
        elif extension == 'mp':
            import msgpack
            sample[key] = msgpack.unpackb(value, raw=False)
        elif extension in ['pt', 'pth']:
            import torch
            sample[key] = torch.load(io.BytesIO(value))
        elif extension in ['pickle', 'pkl']:
            import pickle
            sample[key] = pickle.loads(value)
    return sample