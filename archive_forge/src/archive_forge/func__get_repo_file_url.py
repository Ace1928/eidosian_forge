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
def _get_repo_file_url(namespace, filename):
    """Return the URL for hosted file in Gluon repository.

    Parameters
    ----------
    namespace : str
        Namespace of the file.
    filename : str
        Name of the file
    """
    return '{base_url}{namespace}/{filename}'.format(base_url=_get_repo_url(), namespace=namespace, filename=filename)