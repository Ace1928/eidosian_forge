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
def ExecMergeInfoPlist(self, output, *inputs):
    """Merge multiple .plist files into a single .plist file."""
    merged_plist = {}
    for path in inputs:
        plist = self._LoadPlistMaybeBinary(path)
        self._MergePlist(merged_plist, plist)
    plistlib.writePlist(merged_plist, output)