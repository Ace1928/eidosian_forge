from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import socket
import ssl
from googlecloudsdk.core import config
from googlecloudsdk.core import http
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests as core_requests
from googlecloudsdk.core.diagnostics import check_base
from googlecloudsdk.core.diagnostics import diagnostic_base
from googlecloudsdk.core.diagnostics import http_proxy_setup
import httplib2
import requests
from six.moves import http_client
from six.moves import urllib
import socks
def RunChecks(self):
    if not properties.IsDefaultUniverse():
        return True
    return super().RunChecks()