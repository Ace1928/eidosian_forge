import datetime
import uuid
from stat import S_ISDIR, S_ISLNK
import smbclient
from .. import AbstractFileSystem
from ..utils import infer_storage_options
def _as_unc_path(host, path):
    rpath = path.replace('/', '\\')
    unc = f'\\\\{host}{rpath}'
    return unc