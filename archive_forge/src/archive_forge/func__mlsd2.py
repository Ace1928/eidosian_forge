import os
import sys
import uuid
import warnings
from ftplib import FTP, Error, error_perm
from typing import Any
from ..spec import AbstractBufferedFile, AbstractFileSystem
from ..utils import infer_storage_options, isfilelike
def _mlsd2(ftp, path='.'):
    """
    Fall back to using `dir` instead of `mlsd` if not supported.

    This parses a Linux style `ls -l` response to `dir`, but the response may
    be platform dependent.

    Parameters
    ----------
    ftp: ftplib.FTP
    path: str
        Expects to be given path, but defaults to ".".
    """
    lines = []
    minfo = []
    ftp.dir(path, lines.append)
    for line in lines:
        split_line = line.split()
        if len(split_line) < 9:
            continue
        this = (split_line[-1], {'modify': ' '.join(split_line[5:8]), 'unix.owner': split_line[2], 'unix.group': split_line[3], 'unix.mode': split_line[0], 'size': split_line[4]})
        if 'd' == this[1]['unix.mode'][0]:
            this[1]['type'] = 'dir'
        else:
            this[1]['type'] = 'file'
        minfo.append(this)
    return minfo