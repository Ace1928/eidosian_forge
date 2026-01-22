import gzip
import os
import re
from io import BytesIO
from typing import Type
from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore
from ..objects import Blob
from ..repo import BaseRepo, MemoryRepo
from ..server import DictBackend
from ..web import (
from .utils import make_object, make_tag
def _get_zstream(self, text):
    zstream = BytesIO()
    zfile = gzip.GzipFile(fileobj=zstream, mode='wb')
    zfile.write(text)
    zfile.close()
    zlength = zstream.tell()
    zstream.seek(0)
    return (zstream, zlength)