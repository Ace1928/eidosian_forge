import os
import sys
import hashlib
import uuid
import warnings
import collections
import weakref
import requests
import numpy as np
from .. import ndarray
from ..util import is_np_shape, is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
def _get_repo_url():
    """Return the base URL for Gluon dataset and model repository."""
    default_repo = 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/'
    repo_url = os.environ.get('MXNET_GLUON_REPO', default_repo)
    if repo_url[-1] != '/':
        repo_url = repo_url + '/'
    return repo_url