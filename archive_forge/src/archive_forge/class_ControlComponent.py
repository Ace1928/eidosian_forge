from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
class ControlComponent:
    """Abstract base class for control directory components.

    This provides interfaces that are common across controldirs,
    repositories, branches, and workingtree control directories.

    They all expose two urls and transports: the *user* URL is the
    one that stops above the control directory (eg .bzr) and that
    should normally be used in messages, and the *control* URL is
    under that in eg .bzr/checkout and is used to read the control
    files.

    This can be used as a mixin and is intended to fit with
    foreign formats.
    """

    @property
    def control_transport(self) -> _mod_transport.Transport:
        raise NotImplementedError

    @property
    def control_url(self) -> str:
        return self.control_transport.base

    @property
    def user_transport(self) -> _mod_transport.Transport:
        raise NotImplementedError

    @property
    def user_url(self) -> str:
        return self.user_transport.base
    _format: 'ControlComponentFormat'