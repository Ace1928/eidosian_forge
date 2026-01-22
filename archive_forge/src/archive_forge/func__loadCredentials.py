import os
from typing import Any, Dict, Optional, Tuple, Union
from zope.interface import Attribute, Interface, implementer
from twisted.cred import error
from twisted.cred.credentials import (
from twisted.internet import defer
from twisted.internet.defer import Deferred
from twisted.logger import Logger
from twisted.python import failure
def _loadCredentials(self):
    """
        Loads the credentials from the configured file.

        @return: An iterable of C{username, password} couples.
        @rtype: C{iterable}

        @raise UnauthorizedLogin: when failing to read the credentials from the
            file.
        """
    try:
        with open(self.filename, 'rb') as f:
            for line in f:
                line = line.rstrip()
                parts = line.split(self.delim)
                if self.ufield >= len(parts) or self.pfield >= len(parts):
                    continue
                if self.caseSensitive:
                    yield (parts[self.ufield], parts[self.pfield])
                else:
                    yield (parts[self.ufield].lower(), parts[self.pfield])
    except OSError as e:
        self._log.error('Unable to load credentials db: {e!r}', e=e)
        raise error.UnauthorizedLogin()