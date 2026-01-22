import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
class _CaseSensitiveFilesystemFeature(Feature):

    def _probe(self):
        if CaseInsCasePresFilenameFeature.available():
            return False
        elif CaseInsensitiveFilesystemFeature.available():
            return False
        else:
            return True

    def feature_name(self):
        return 'case-sensitive filesystem'