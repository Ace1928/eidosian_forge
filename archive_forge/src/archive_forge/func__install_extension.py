import base64
import copy
import json
import os
import re
import shutil
import sys
import tempfile
import warnings
import zipfile
from io import BytesIO
from xml.dom import minidom
from typing_extensions import deprecated
from selenium.common.exceptions import WebDriverException
@deprecated('Addons must be added after starting the session')
def _install_extension(self, addon, unpack=True):
    """Installs addon from a filepath, url or directory of addons in the
        profile.

        - path: url, absolute path to .xpi, or directory of addons
        - unpack: whether to unpack unless specified otherwise in the install.rdf
        """
    tmpdir = None
    xpifile = None
    if addon.endswith('.xpi'):
        tmpdir = tempfile.mkdtemp(suffix='.' + os.path.split(addon)[-1])
        compressed_file = zipfile.ZipFile(addon, 'r')
        for name in compressed_file.namelist():
            if name.endswith('/'):
                if not os.path.isdir(os.path.join(tmpdir, name)):
                    os.makedirs(os.path.join(tmpdir, name))
            else:
                if not os.path.isdir(os.path.dirname(os.path.join(tmpdir, name))):
                    os.makedirs(os.path.dirname(os.path.join(tmpdir, name)))
                data = compressed_file.read(name)
                with open(os.path.join(tmpdir, name), 'wb') as f:
                    f.write(data)
        xpifile = addon
        addon = tmpdir
    addon_details = self._addon_details(addon)
    addon_id = addon_details.get('id')
    assert addon_id, f'The addon id could not be found: {addon}'
    extensions_dir = os.path.join(self._profile_dir, 'extensions')
    addon_path = os.path.join(extensions_dir, addon_id)
    if not unpack and (not addon_details['unpack']) and xpifile:
        if not os.path.exists(extensions_dir):
            os.makedirs(extensions_dir)
            os.chmod(extensions_dir, 493)
        shutil.copy(xpifile, addon_path + '.xpi')
    elif not os.path.exists(addon_path):
        shutil.copytree(addon, addon_path, symlinks=True)
    if tmpdir:
        shutil.rmtree(tmpdir)