import datetime
import uuid
from stat import S_ISDIR, S_ISLNK
import smbclient
from .. import AbstractFileSystem
from ..utils import infer_storage_options
Remove the temp file on failure.