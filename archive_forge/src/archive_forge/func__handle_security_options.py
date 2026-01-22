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
def _handle_security_options(options: _CaseInsensitiveDictionary) -> _CaseInsensitiveDictionary:
    """Raise appropriate errors when conflicting TLS options are present in
    the options dictionary.

    :Parameters:
        - `options`: Instance of _CaseInsensitiveDictionary containing
          MongoDB URI options.
    """
    tlsinsecure = options.get('tlsinsecure')
    if tlsinsecure is not None:
        for opt in _IMPLICIT_TLSINSECURE_OPTS:
            if opt in options:
                err_msg = 'URI options %s and %s cannot be specified simultaneously.'
                raise InvalidURI(err_msg % (options.cased_key('tlsinsecure'), options.cased_key(opt)))
    tlsallowinvalidcerts = options.get('tlsallowinvalidcertificates')
    if tlsallowinvalidcerts is not None:
        if 'tlsdisableocspendpointcheck' in options:
            err_msg = 'URI options %s and %s cannot be specified simultaneously.'
            raise InvalidURI(err_msg % ('tlsallowinvalidcertificates', options.cased_key('tlsdisableocspendpointcheck')))
        if tlsallowinvalidcerts is True:
            options['tlsdisableocspendpointcheck'] = True
    tlscrlfile = options.get('tlscrlfile')
    if tlscrlfile is not None:
        for opt in ('tlsinsecure', 'tlsallowinvalidcertificates', 'tlsdisableocspendpointcheck'):
            if options.get(opt) is True:
                err_msg = 'URI option %s=True cannot be specified when CRL checking is enabled.'
                raise InvalidURI(err_msg % (opt,))
    if 'ssl' in options and 'tls' in options:

        def truth_value(val: Any) -> Any:
            if val in ('true', 'false'):
                return val == 'true'
            if isinstance(val, bool):
                return val
            return val
        if truth_value(options.get('ssl')) != truth_value(options.get('tls')):
            err_msg = 'Can not specify conflicting values for URI options %s and %s.'
            raise InvalidURI(err_msg % (options.cased_key('ssl'), options.cased_key('tls')))
    return options