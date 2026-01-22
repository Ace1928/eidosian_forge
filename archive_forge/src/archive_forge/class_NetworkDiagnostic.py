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
class NetworkDiagnostic(diagnostic_base.Diagnostic):
    """Diagnose and fix local network connection issues."""

    def __init__(self):
        intro = 'Network diagnostic detects and fixes local network connection issues.'
        super(NetworkDiagnostic, self).__init__(intro=intro, title='Network diagnostic', checklist=[ReachabilityChecker()])

    def RunChecks(self):
        if not properties.IsDefaultUniverse():
            return True
        return super().RunChecks()