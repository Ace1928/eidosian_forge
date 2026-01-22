import os
import posixpath
import sys
import urllib.parse
import warnings
from os.path import join as pjoin
import pyarrow as pa
from pyarrow.util import doc, _stringify_path, _is_path_like, _DEPR_MSG
def _sanitize_s3(path):
    if path.startswith('s3://'):
        return path.replace('s3://', '')
    else:
        return path