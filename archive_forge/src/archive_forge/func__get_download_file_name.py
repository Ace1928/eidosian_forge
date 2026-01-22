import logging
import os
import tarfile
import warnings
import zipfile
from . import _constants as C
from . import vocab
from ... import ndarray as nd
from ... import registry
from ... import base
from ...util import is_np_array
from ... import numpy as _mx_np
from ... import numpy_extension as _mx_npx
@classmethod
def _get_download_file_name(cls, pretrained_file_name):
    return '.'.join(pretrained_file_name.split('.')[:-1]) + '.zip'