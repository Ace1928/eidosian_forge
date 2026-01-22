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
class ArchiveFileListIterator(collections.abc.Iterator):

    def __init__(self, archive_file_list):
        self._archive_file_list = archive_file_list
        self._index = 0

    def __next__(self) -> ArchiveFile:
        if self._index == len(self._archive_file_list):
            raise StopIteration
        res = self._archive_file_list[self._index]
        self._index += 1
        return res