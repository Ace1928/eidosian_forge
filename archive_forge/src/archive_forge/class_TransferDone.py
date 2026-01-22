import os
import sys
import uuid
import warnings
from ftplib import FTP, Error, error_perm
from typing import Any
from ..spec import AbstractBufferedFile, AbstractFileSystem
from ..utils import infer_storage_options, isfilelike
class TransferDone(Exception):
    """Internal exception to break out of transfer"""
    pass