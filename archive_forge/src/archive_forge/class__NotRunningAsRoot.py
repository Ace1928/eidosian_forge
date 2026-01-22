import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
class _NotRunningAsRoot(Feature):

    def _probe(self):
        try:
            uid = os.getuid()
        except AttributeError:
            return True
        return uid != 0

    def feature_name(self):
        return 'Not running as root'