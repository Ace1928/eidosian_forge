import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from ... import bedding
from ... import branch as _mod_branch
from ... import controldir, errors, hooks, urlutils
from ... import version_string as breezy_version
from ...config import AuthenticationConfig, GlobalStack
from ...errors import (InvalidHttpResponse, PermissionDenied,
from ...forge import (Forge, ForgeLoginRequired, MergeProposal,
from ...git.urls import git_url_to_bzr_url
from ...i18n import gettext
from ...trace import note
from ...transport import get_transport
from ...transport.http import default_user_agent
class _LazyDict(dict):

    def __init__(self, base, load_fn):
        self._load_fn = load_fn
        super().update(base)

    def _load_full(self):
        super().update(self._load_fn())
        self._load_fn = None

    def __getitem__(self, key):
        if self._load_fn is not None:
            try:
                return super().__getitem__(key)
            except KeyError:
                self._load_full()
        return super().__getitem__(key)

    def items(self):
        self._load_full()
        return super().items()

    def keys(self):
        self._load_full()
        return super().keys()

    def values(self):
        self._load_full()
        return super().values()

    def __contains__(self, key):
        if super().__contains__(key):
            return True
        if self._load_fn is not None:
            self._load_full()
            return super().__contains__(key)
        return False

    def __delitem__(self, name):
        raise NotImplementedError

    def __setitem__(self, name, value):
        raise NotImplementedError

    def get(self, name, default=None):
        if self._load_fn is not None:
            try:
                return super().get(name, default)
            except KeyError:
                self._load_full()
        return super().get(name, default)

    def pop(self):
        raise NotImplementedError

    def popitem(self):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError