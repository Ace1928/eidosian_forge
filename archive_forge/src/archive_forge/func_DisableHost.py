from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import utils
def DisableHost(self, project):
    """Disable the project with the given ID as an XPN host."""
    request_tuple = (self.client.projects, 'DisableXpnHost', self.messages.ComputeProjectsDisableXpnHostRequest(project=project))
    msg = 'disable [{project}] as XPN host'.format(project=project)
    self._MakeRequestSync(request_tuple, msg)