from .parseVertexGramMatrixFile import *
from .verificationError import *
from .orb import __path__ as orb_path
from snappy.snap.t3mlite import Mcomplex
import subprocess
import tempfile
import shutil
import os
class TmpDir:
    """
    To be used in a with statement, creating a temporary
    directory deleted at the end of the with statement.
    """

    def __init__(self, delete=True):
        self.delete = delete

    def __enter__(self):
        self.path = tempfile.mkdtemp()
        return self

    def __exit__(self, type, value, traceback):
        if self.delete:
            shutil.rmtree(self.path)