import os
import shutil
from tempfile import template, mkdtemp
class InTemporaryDirectory(TemporaryDirectory):

    def __enter__(self):
        self._pwd = os.getcwd()
        os.chdir(self.name)
        return super(InTemporaryDirectory, self).__enter__()

    def __exit__(self, exc, value, tb):
        os.chdir(self._pwd)
        return super(InTemporaryDirectory, self).__exit__(exc, value, tb)