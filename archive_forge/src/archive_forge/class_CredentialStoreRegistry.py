import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
class CredentialStoreRegistry(registry.Registry):
    """A class that registers credential stores.

    A credential store provides access to credentials via the password_encoding
    field in authentication.conf sections.

    Except for stores provided by brz itself, most stores are expected to be
    provided by plugins that will therefore use
    register_lazy(password_encoding, module_name, member_name, help=help,
    fallback=fallback) to install themselves.

    A fallback credential store is one that is queried if no credentials can be
    found via authentication.conf.
    """

    def get_credential_store(self, encoding=None):
        cs = self.get(encoding)
        if callable(cs):
            cs = cs()
        return cs

    def is_fallback(self, name):
        """Check if the named credentials store should be used as fallback."""
        return self.get_info(name)

    def get_fallback_credentials(self, scheme, host, port=None, user=None, path=None, realm=None):
        """Request credentials from all fallback credentials stores.

        The first credentials store that can provide credentials wins.
        """
        credentials = None
        for name in self.keys():
            if not self.is_fallback(name):
                continue
            cs = self.get_credential_store(name)
            credentials = cs.get_credentials(scheme, host, port, user, path, realm)
            if credentials is not None:
                break
        return credentials

    def register(self, key, obj, help=None, override_existing=False, fallback=False):
        """Register a new object to a name.

        Args:
          key: This is the key to use to request the object later.
          obj: The object to register.
          help: Help text for this entry. This may be a string or
                a callable. If it is a callable, it should take two
                parameters (registry, key): this registry and the key that
                the help was registered under.
          override_existing: Raise KeyErorr if False and something has
                already been registered for that key. If True, ignore if there
                is an existing key (always register the new value).
          fallback: Whether this credential store should be
                used as fallback.
        """
        return super().register(key, obj, help, info=fallback, override_existing=override_existing)

    def register_lazy(self, key, module_name, member_name, help=None, override_existing=False, fallback=False):
        """Register a new credential store to be loaded on request.

        Args:
          module_name: The python path to the module. Such as 'os.path'.
          member_name: The member of the module to return.  If empty or
                None, get() will return the module itself.
          help: Help text for this entry. This may be a string or
                a callable.
          override_existing: If True, replace the existing object
                with the new one. If False, if there is already something
                registered with the same key, raise a KeyError
          fallback: Whether this credential store should be
                used as fallback.
        """
        return super().register_lazy(key, module_name, member_name, help, info=fallback, override_existing=override_existing)