import os
import stat
import textwrap
from email import message_from_file
from email.message import Message
from tempfile import NamedTemporaryFile
from typing import Optional, List
from distutils.util import rfc822_escape
from . import _normalization, _reqs
from .extern.packaging.markers import Marker
from .extern.packaging.requirements import Requirement
from .extern.packaging.version import Version
from .warnings import SetuptoolsDeprecationWarning
def write_pkg_info(self, base_dir):
    """Write the PKG-INFO file into the release tree."""
    temp = ''
    final = os.path.join(base_dir, 'PKG-INFO')
    try:
        with NamedTemporaryFile('w', encoding='utf-8', dir=base_dir, delete=False) as f:
            temp = f.name
            self.write_pkg_file(f)
        permissions = stat.S_IMODE(os.lstat(temp).st_mode)
        os.chmod(temp, permissions | stat.S_IRGRP | stat.S_IROTH)
        os.replace(temp, final)
    finally:
        if temp and os.path.exists(temp):
            os.remove(temp)