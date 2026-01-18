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
Expands variables "$(variable)" in data.

    Args:
      data: object, can be either string, list or dictionary
      substitutions: dictionary, variable substitutions to perform

    Returns:
      Copy of data where each references to "$(variable)" has been replaced
      by the corresponding value found in substitutions, or left intact if
      the key was not found.
    