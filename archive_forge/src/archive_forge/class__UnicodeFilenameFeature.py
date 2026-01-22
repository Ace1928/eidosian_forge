import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
class _UnicodeFilenameFeature(Feature):
    """Does the filesystem support Unicode filenames?"""

    def _probe(self):
        try:
            os.stat('α⠿')
        except UnicodeEncodeError:
            return False
        except OSError:
            return True
        else:
            return True