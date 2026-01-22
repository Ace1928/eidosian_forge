from __future__ import annotations
import re
import sys
import warnings
from typing import (
from urllib.parse import unquote_plus
from pymongo.client_options import _parse_ssl_options
from pymongo.common import (
from pymongo.errors import ConfigurationError, InvalidURI
from pymongo.srv_resolver import _HAVE_DNSPYTHON, _SrvResolver
from pymongo.typings import _Address
def _handle_option_deprecations(options: _CaseInsensitiveDictionary) -> _CaseInsensitiveDictionary:
    """Issue appropriate warnings when deprecated options are present in the
    options dictionary. Removes deprecated option key, value pairs if the
    options dictionary is found to also have the renamed option.

    :Parameters:
        - `options`: Instance of _CaseInsensitiveDictionary containing
          MongoDB URI options.
    """
    for optname in list(options):
        if optname in URI_OPTIONS_DEPRECATION_MAP:
            mode, message = URI_OPTIONS_DEPRECATION_MAP[optname]
            if mode == 'renamed':
                newoptname = message
                if newoptname in options:
                    warn_msg = "Deprecated option '%s' ignored in favor of '%s'."
                    warnings.warn(warn_msg % (options.cased_key(optname), options.cased_key(newoptname)), DeprecationWarning, stacklevel=2)
                    options.pop(optname)
                    continue
                warn_msg = "Option '%s' is deprecated, use '%s' instead."
                warnings.warn(warn_msg % (options.cased_key(optname), newoptname), DeprecationWarning, stacklevel=2)
            elif mode == 'removed':
                warn_msg = "Option '%s' is deprecated. %s."
                warnings.warn(warn_msg % (options.cased_key(optname), message), DeprecationWarning, stacklevel=2)
    return options