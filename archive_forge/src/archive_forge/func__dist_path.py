import subprocess, sys, os
from distutils.core import Command
from distutils.debug import DEBUG
from distutils.file_util import write_file
from distutils.errors import *
from distutils.sysconfig import get_python_version
from distutils import log
def _dist_path(self, path):
    return os.path.join(self.dist_dir, os.path.basename(path))