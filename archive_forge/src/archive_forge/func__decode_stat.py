import datetime
import logging
import os
import types
import uuid
from stat import S_ISDIR, S_ISLNK
import paramiko
from .. import AbstractFileSystem
from ..utils import infer_storage_options
@staticmethod
def _decode_stat(stat, parent_path=None):
    if S_ISDIR(stat.st_mode):
        t = 'directory'
    elif S_ISLNK(stat.st_mode):
        t = 'link'
    else:
        t = 'file'
    out = {'name': '', 'size': stat.st_size, 'type': t, 'uid': stat.st_uid, 'gid': stat.st_gid, 'time': datetime.datetime.fromtimestamp(stat.st_atime, tz=datetime.timezone.utc), 'mtime': datetime.datetime.fromtimestamp(stat.st_mtime, tz=datetime.timezone.utc)}
    if parent_path:
        out['name'] = '/'.join([parent_path.rstrip('/'), stat.filename])
    return out