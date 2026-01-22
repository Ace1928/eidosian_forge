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
def BrowseApp(project, service, version, launch_browser):
    """Let you browse the given service at the given version.

  Args:
    project: str, project ID.
    service: str, specific service, 'default' if None
    version: str, specific version, latest if None
    launch_browser: boolean, if False only print url

  Returns:
    None if the browser should open the URL
    The relevant output as a dict for calliope format to print if not

  Raises:
    MissingApplicationError: If an app does not exist.
  """
    try:
        url = deploy_command_util.GetAppHostname(app_id=project, service=service, version=version, use_ssl=appinfo.SECURE_HTTPS, deploy=False)
    except apitools_exceptions.HttpNotFoundError:
        log.debug('No app found:', exc_info=True)
        raise exceptions.MissingApplicationError(project)
    if check_browser.ShouldLaunchBrowser(launch_browser):
        OpenURL(url)
        return None
    else:
        return {'url': url, 'service': service or 'default', 'version': version}