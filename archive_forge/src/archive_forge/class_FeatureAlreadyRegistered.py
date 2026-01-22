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
class FeatureAlreadyRegistered(errors.BzrError):
    _fmt = 'The feature %(feature)s has already been registered.'

    def __init__(self, feature):
        self.feature = feature