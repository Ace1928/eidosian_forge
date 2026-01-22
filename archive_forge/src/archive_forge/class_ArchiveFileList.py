import collections.abc
import contextlib
import datetime
import errno
import functools
import io
import os
import pathlib
import queue
import re
import stat
import sys
import time
from multiprocessing import Process
from threading import Thread
from typing import IO, Any, BinaryIO, Collection, Dict, List, Optional, Tuple, Type, Union
import multivolumefile
from py7zr.archiveinfo import Folder, Header, SignatureHeader
from py7zr.callbacks import ExtractCallback
from py7zr.compressor import SupportedMethods, get_methods_names
from py7zr.exceptions import (
from py7zr.helpers import (
from py7zr.properties import DEFAULT_FILTERS, FILTER_DEFLATE64, MAGIC_7Z, get_default_blocksize, get_memory_limit
class ArchiveFileList(collections.abc.Iterable):
    """Iteratable container of ArchiveFile."""

    def __init__(self, offset: int=0):
        self.files_list: List[dict] = []
        self.index = 0
        self.offset = offset

    def append(self, file_info: Dict[str, Any]) -> None:
        self.files_list.append(file_info)

    def __len__(self) -> int:
        return len(self.files_list)

    def __iter__(self) -> 'ArchiveFileListIterator':
        return ArchiveFileListIterator(self)

    def __getitem__(self, index):
        if index > len(self.files_list):
            raise IndexError
        if index < 0:
            raise IndexError
        res = ArchiveFile(index + self.offset, self.files_list[index])
        return res