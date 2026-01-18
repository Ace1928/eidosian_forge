import glob
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from os.path import join as pjoin
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import pytest
from jupyter_core import paths
from jupyterlab import commands
from jupyterlab.commands import (
from jupyterlab.coreconfig import CoreConfig, _get_default_core_data
@pytest.mark.slow
def test_uninstall_core_extension(self):
    assert uninstall_extension('@jupyterlab/console-extension') is True
    app_dir = self.app_dir
    build()
    with open(pjoin(app_dir, 'staging', 'package.json')) as fid:
        data = json.load(fid)
    extensions = data['jupyterlab']['extensions']
    assert '@jupyterlab/console-extension' not in extensions
    assert not check_extension('@jupyterlab/console-extension')
    assert install_extension('@jupyterlab/console-extension') is True
    build()
    with open(pjoin(app_dir, 'staging', 'package.json')) as fid:
        data = json.load(fid)
    extensions = data['jupyterlab']['extensions']
    assert '@jupyterlab/console-extension' in extensions
    assert check_extension('@jupyterlab/console-extension')