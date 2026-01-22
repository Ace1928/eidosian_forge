from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import deploy_command_util
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.command_lib.util import check_browser
from googlecloudsdk.core import log
from googlecloudsdk.core.credentials import devshell
from googlecloudsdk.third_party.appengine.api import appinfo
def OpenURL(url):
    """Open a URL in the default web browser in a new tab.

  Args:
    url: The full HTTP(S) URL to open.
  """
    import webbrowser
    if not devshell.IsDevshellEnvironment():
        log.status.Print('Opening [{0}] in a new tab in your default browser.'.format(url))
    webbrowser.open_new_tab(url)