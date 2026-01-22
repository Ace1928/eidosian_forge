import collections
import email
from email.mime import multipart
from email.mime import text
import os
import pkgutil
import string
from urllib import parse as urlparse
from neutronclient.common import exceptions as q_exceptions
from novaclient import api_versions
from novaclient import client as nc
from novaclient import exceptions
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import netutils
import tenacity
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_exception
from heat.engine.clients import client_plugin
from heat.engine.clients import microversion_mixin
from heat.engine.clients import os as os_client
from heat.engine import constraints
class ConsoleUrls(collections.abc.Mapping):

    def __init__(self, server):
        self.console_method = server.get_console_url
        self.support_console_types = ['novnc', 'xvpvnc', 'spice-html5', 'rdp-html5', 'serial', 'webmks']

    def __getitem__(self, key):
        try:
            if key not in self.support_console_types:
                raise exceptions.UnsupportedConsoleType(key)
            if key == 'webmks':
                data = nc().servers.get_console_url(server, key)
            else:
                data = self.console_method(key)
            console_data = data.get('remote_console', data.get('console'))
            url = console_data['url']
        except exceptions.UnsupportedConsoleType as ex:
            url = ex.message
        except Exception as e:
            url = _('Cannot get console url: %s') % str(e)
        return url

    def __len__(self):
        return len(self.support_console_types)

    def __iter__(self):
        return (key for key in self.support_console_types)