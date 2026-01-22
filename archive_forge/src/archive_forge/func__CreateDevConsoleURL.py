from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import browser_dispatcher
from googlecloudsdk.core import properties
from six.moves import urllib
def _CreateDevConsoleURL(project, service='default', version=None, logs=False):
    """Creates a URL to a page in the Developer Console according to the params.

  Args:
    project: The app engine project id
    service: A service belonging to the project
    version: A version belonging to the service, or all versions if omitted
    logs: If true, go to the log section instead of the dashboard
  Returns:
    The URL to the respective page in the Developer Console
  """
    if service is None:
        service = 'default'
    query = [('project', project), ('serviceId', service)]
    if version:
        query.append(('versionId', version))
    query_string = urllib.parse.urlencode(query)
    return (LOGS_URL if logs else CONSOLE_URL).format(query=query_string)