import posixpath
import stat
import struct
import tarfile
from contextlib import closing
from io import BytesIO
from os import SEEK_END
Recursively walk a dulwich Tree, yielding tuples of
    (absolute path, TreeEntry) along the way.
    