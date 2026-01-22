import datetime
import uuid
from stat import S_ISDIR, S_ISLNK
import smbclient
from .. import AbstractFileSystem
from ..utils import infer_storage_options
@staticmethod
def _get_kwargs_from_urls(path):
    out = infer_storage_options(path)
    out.pop('path', None)
    out.pop('protocol', None)
    return out