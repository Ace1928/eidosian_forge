import contextlib
import sys
from typing import TYPE_CHECKING, Set, cast
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import config, controldir, errors, lockdir
from .. import transport as _mod_transport
from ..trace import mutter, note, warning
from ..transport import do_catching_redirections, local
def _read_branch_list(self):
    """Read the branch list.

        :return: List of branch names.
        """
    try:
        f = self.control_transport.get('branch-list')
    except _mod_transport.NoSuchFile:
        return []
    ret = []
    try:
        for name in f:
            ret.append(name.rstrip(b'\n').decode('utf-8'))
    finally:
        f.close()
    return ret