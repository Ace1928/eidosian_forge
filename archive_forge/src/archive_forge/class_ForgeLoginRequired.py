import re
from typing import Optional, Type
from . import errors, hooks, registry, urlutils
class ForgeLoginRequired(errors.BzrError):
    """Action requires forge login credentials."""
    _fmt = 'Action requires credentials for hosting site %(forge)r.'

    def __init__(self, forge):
        errors.BzrError.__init__(self)
        self.forge = forge