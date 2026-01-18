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
def test_uninstall_all_extensions(self):
    install_extension(self.mock_extension)
    install_extension(self.mock_mimeextension)
    ext_name = self.pkg_names['extension']
    mime_ext_name = self.pkg_names['mimeextension']
    assert check_extension(ext_name) is True
    assert check_extension(mime_ext_name) is True
    assert uninstall_extension(all_=True) is True
    extensions = get_app_info()['extensions']
    assert ext_name not in extensions
    assert mime_ext_name not in extensions