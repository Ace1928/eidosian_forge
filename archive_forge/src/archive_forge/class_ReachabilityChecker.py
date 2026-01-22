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
class ReachabilityChecker(check_base.Checker):
    """Checks whether the hosts of given urls are reachable."""

    @property
    def issue(self):
        return 'network connection'

    def Check(self, urls=None, first_run=True):
        """Run reachability check.

    Args:
      urls: iterable(str), The list of urls to check connection to. Defaults to
        DefaultUrls() (above) if not supplied.
      first_run: bool, True if first time this has been run this invocation.

    Returns:
      A tuple of (check_base.Result, fixer) where fixer is a function that can
        be used to fix a failed check, or  None if the check passed or failed
        with no applicable fix.
    """
        if urls is None:
            urls = DefaultUrls()
        failures = []
        for url in urls:
            fail = CheckURLHttplib2(url)
            if fail:
                failures.append(fail)
        for url in urls:
            fail = CheckURLRequests(url)
            if fail:
                failures.append(fail)
        if failures:
            fail_message = ConstructMessageFromFailures(failures, first_run)
            result = check_base.Result(passed=False, message=fail_message, failures=failures)
            fixer = http_proxy_setup.ChangeGcloudProxySettings
            return (result, fixer)
        pass_message = 'Reachability Check {0}.'.format('passed' if first_run else 'now passes')
        result = check_base.Result(passed=True, message='No URLs to check.' if not urls else pass_message)
        return (result, None)