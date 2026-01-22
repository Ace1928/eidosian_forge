import fcntl
import fnmatch
import glob
import json
import os
import plistlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
def _ExpandVariables(self, data, substitutions):
    """Expands variables "$(variable)" in data.

    Args:
      data: object, can be either string, list or dictionary
      substitutions: dictionary, variable substitutions to perform

    Returns:
      Copy of data where each references to "$(variable)" has been replaced
      by the corresponding value found in substitutions, or left intact if
      the key was not found.
    """
    if isinstance(data, str):
        for key, value in substitutions.items():
            data = data.replace('$(%s)' % key, value)
        return data
    if isinstance(data, list):
        return [self._ExpandVariables(v, substitutions) for v in data]
    if isinstance(data, dict):
        return {k: self._ExpandVariables(data[k], substitutions) for k in data}
    return data