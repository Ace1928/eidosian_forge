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
def ExecPackageFramework(self, framework, version):
    """Takes a path to Something.framework and the Current version of that and
    sets up all the symlinks."""
    binary = os.path.basename(framework).split('.')[0]
    CURRENT = 'Current'
    RESOURCES = 'Resources'
    VERSIONS = 'Versions'
    if not os.path.exists(os.path.join(framework, VERSIONS, version, binary)):
        return
    pwd = os.getcwd()
    os.chdir(framework)
    self._Relink(version, os.path.join(VERSIONS, CURRENT))
    self._Relink(os.path.join(VERSIONS, CURRENT, binary), binary)
    self._Relink(os.path.join(VERSIONS, CURRENT, RESOURCES), RESOURCES)
    os.chdir(pwd)