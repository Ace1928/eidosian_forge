import errno
from eventlet.green import socket
import functools
import os
import re
import urllib
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import netutils
from oslo_utils import strutils
from webob import exc
from glance.common import exception
from glance.common import location_strategy
from glance.common import timeutils
from glance.common import wsgi
from glance.i18n import _, _LE, _LW
def validate_import_uri(uri):
    """Validate requested uri for Image Import web-download.

    :param uri: target uri to be validated
    """
    if not uri:
        return False
    parsed_uri = urllib.parse.urlparse(uri)
    scheme = parsed_uri.scheme
    host = parsed_uri.hostname
    port = parsed_uri.port
    wl_schemes = CONF.import_filtering_opts.allowed_schemes
    bl_schemes = CONF.import_filtering_opts.disallowed_schemes
    wl_hosts = CONF.import_filtering_opts.allowed_hosts
    bl_hosts = CONF.import_filtering_opts.disallowed_hosts
    wl_ports = CONF.import_filtering_opts.allowed_ports
    bl_ports = CONF.import_filtering_opts.disallowed_ports
    if wl_schemes and bl_schemes:
        bl_schemes = []
        LOG.debug('Both allowed and disallowed schemes has been configured. Will only process allowed list.')
    if wl_hosts and bl_hosts:
        bl_hosts = []
        LOG.debug('Both allowed and disallowed hosts has been configured. Will only process allowed list.')
    if wl_ports and bl_ports:
        bl_ports = []
        LOG.debug('Both allowed and disallowed ports has been configured. Will only process allowed list.')
    if not scheme or (wl_schemes and scheme not in wl_schemes or parsed_uri.scheme in bl_schemes):
        return False
    if not host or (wl_hosts and host not in wl_hosts or host in bl_hosts):
        return False
    if port and (wl_ports and port not in wl_ports or port in bl_ports):
        return False
    return True