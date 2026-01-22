import io
import json
import os.path
import pickle
import tempfile
import torch
from torch.utils.data.datapipes.utils.common import StreamWrapper
class MatHandler:

    def __init__(self, **loadmat_kwargs) -> None:
        try:
            import scipy.io as sio
        except ImportError as e:
            raise ModuleNotFoundError('Package `scipy` is required to be installed for mat file.Please use `pip install scipy` or `conda install scipy`to install the package') from e
        self.sio = sio
        self.loadmat_kwargs = loadmat_kwargs

    def __call__(self, extension, data):
        if extension != 'mat':
            return None
        with io.BytesIO(data) as stream:
            return self.sio.loadmat(stream, **self.loadmat_kwargs)