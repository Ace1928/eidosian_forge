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
class ZipFileWrapper:

    def __init__(self, fileobj, mode):
        self.zipfile = zipfile.ZipFile(fileobj, mode)

    def getmembers(self):
        for info in self.zipfile.infolist():
            yield ZipInfoWrapper(self.zipfile, info)

    def extractfile(self, infowrapper):
        return BytesIO(self.zipfile.read(infowrapper.name))

    def add(self, filename):
        if isdir(filename):
            self.zipfile.writestr(filename + '/', '')
        else:
            self.zipfile.write(filename)

    def close(self):
        self.zipfile.close()