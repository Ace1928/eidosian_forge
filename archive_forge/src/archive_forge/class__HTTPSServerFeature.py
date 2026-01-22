import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
class _HTTPSServerFeature(Feature):
    """Some tests want an https Server, check if one is available.
    """

    def _probe(self):
        try:
            import ssl
            return True
        except ModuleNotFoundError:
            return False

    def feature_name(self):
        return 'HTTPSServer'