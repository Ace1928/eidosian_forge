import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
class ExecutableFeature(Feature):
    """Feature testing whether an executable of a given name is on the PATH."""

    def __init__(self, name):
        super().__init__()
        self.name = name
        self._path = None

    @property
    def path(self):
        self.available()
        return self._path

    def _probe(self):
        self._path = osutils.find_executable_on_path(self.name)
        return self._path is not None

    def feature_name(self):
        return '%s executable' % self.name