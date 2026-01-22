import errno
import os
import re
import stat
import tarfile
import zipfile
from io import BytesIO
from . import urlutils
from .bzr import generate_ids
from .controldir import ControlDir, is_control_filename
from .errors import BzrError, CommandError, NotBranchError
from .osutils import (basename, file_iterator, file_kind, isdir, pathjoin,
from .trace import warning
from .transform import resolve_conflicts
from .transport import NoSuchFile, get_transport
from .workingtree import WorkingTree
class DirWrapper:

    def __init__(self, fileobj, mode='r'):
        if mode != 'r':
            raise AssertionError('only readonly supported')
        self.root = os.path.realpath(fileobj.read().decode('utf-8'))

    def __repr__(self):
        return 'DirWrapper(%r)' % self.root

    def getmembers(self, subdir=None):
        if subdir is not None:
            mydir = pathjoin(self.root, subdir)
        else:
            mydir = self.root
        for child in os.listdir(mydir):
            if subdir is not None:
                child = pathjoin(subdir, child)
            fi = FileInfo(self.root, child)
            yield fi
            if fi.isdir():
                yield from self.getmembers(child)

    def extractfile(self, member):
        return open(member.fullpath, 'rb')