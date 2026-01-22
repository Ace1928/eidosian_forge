import abc
import codecs
import os.path
import random
import re
import sys
import uuid
import weakref
import ldap.controls
import ldap.filter
import ldappool
from oslo_log import log
from oslo_utils import reflection
from keystone.common import driver_hints
from keystone import exception
from keystone.i18n import _
def _common_ldap_initialization(url, use_tls=False, tls_cacertfile=None, tls_cacertdir=None, tls_req_cert=None, debug_level=None, timeout=None):
    """LDAP initialization for PythonLDAPHandler and PooledLDAPHandler."""
    LOG.debug('LDAP init: url=%s', url)
    LOG.debug('LDAP init: use_tls=%s tls_cacertfile=%s tls_cacertdir=%s tls_req_cert=%s tls_avail=%s', use_tls, tls_cacertfile, tls_cacertdir, tls_req_cert, ldap.TLS_AVAIL)
    if debug_level is not None:
        ldap.set_option(ldap.OPT_DEBUG_LEVEL, debug_level)
    using_ldaps = url.lower().startswith('ldaps')
    if timeout is not None and timeout > 0:
        ldap.set_option(ldap.OPT_NETWORK_TIMEOUT, timeout)
    if use_tls and using_ldaps:
        raise AssertionError(_('Invalid TLS / LDAPS combination'))
    if use_tls or using_ldaps:
        if not ldap.TLS_AVAIL:
            raise ValueError(_('Invalid LDAP TLS_AVAIL option: %s. TLS not available') % ldap.TLS_AVAIL)
        if not tls_cacertfile and (not tls_cacertdir):
            raise ValueError(_('You need to set tls_cacertfile or tls_cacertdir if use_tls is true or url uses ldaps: scheme.'))
        if tls_cacertfile:
            if not os.path.isfile(tls_cacertfile):
                raise IOError(_('tls_cacertfile %s not found or is not a file') % tls_cacertfile)
            ldap.set_option(ldap.OPT_X_TLS_CACERTFILE, tls_cacertfile)
        elif tls_cacertdir:
            if not os.path.isdir(tls_cacertdir):
                raise IOError(_('tls_cacertdir %s not found or is not a directory') % tls_cacertdir)
            ldap.set_option(ldap.OPT_X_TLS_CACERTDIR, tls_cacertdir)
        if tls_req_cert in list(LDAP_TLS_CERTS.values()):
            ldap.set_option(ldap.OPT_X_TLS_REQUIRE_CERT, tls_req_cert)
        else:
            LOG.debug('LDAP TLS: invalid TLS_REQUIRE_CERT Option=%s', tls_req_cert)