import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
class _ChownFeature(Feature):
    """os.chown is supported"""

    def _probe(self):
        return os.name == 'posix' and hasattr(os, 'chown')