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
def test_update_all(self):
    updated = []

    def _mock_update(self, name, *args, **kwargs):
        updated.append(name[0] + name[1:].split('@')[0])
        return True
    original_app_info = commands._AppHandler._get_app_info

    def _mock_app_info(self):
        info = original_app_info(self)
        info['local_extensions'] = []
        return info
    assert install_extension(self.mock_extension) is True
    assert install_extension(self.mock_mimeextension) is True
    p1 = patch.object(commands._AppHandler, '_update_extension', _mock_update)
    p2 = patch.object(commands._AppHandler, '_get_app_info', _mock_app_info)
    with p1, p2:
        assert update_extension(None, all_=True) is True
    assert sorted(updated) == [self.pkg_names['extension'], self.pkg_names['mimeextension']]