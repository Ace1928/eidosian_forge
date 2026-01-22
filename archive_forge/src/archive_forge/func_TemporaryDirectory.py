import asyncio
from functools import partial, singledispatch
from io import BufferedRandom, BufferedReader, BufferedWriter, FileIO, TextIOBase
from tempfile import NamedTemporaryFile as syncNamedTemporaryFile
from tempfile import SpooledTemporaryFile as syncSpooledTemporaryFile
from tempfile import TemporaryDirectory as syncTemporaryDirectory
from tempfile import TemporaryFile as syncTemporaryFile
from tempfile import _TemporaryFileWrapper as syncTemporaryFileWrapper
from ..base import AiofilesContextManager
from ..threadpool.binary import AsyncBufferedIOBase, AsyncBufferedReader, AsyncFileIO
from ..threadpool.text import AsyncTextIOWrapper
from .temptypes import AsyncSpooledTemporaryFile, AsyncTemporaryDirectory
import sys
def TemporaryDirectory(suffix=None, prefix=None, dir=None, loop=None, executor=None):
    """Async open a temporary directory"""
    return AiofilesContextManagerTempDir(_temporary_directory(suffix=suffix, prefix=prefix, dir=dir, loop=loop, executor=executor))