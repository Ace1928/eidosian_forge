from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import utils
def EnableHost(self, project):
    """Enable the project with the given ID as an XPN host."""
    request_tuple = (self.client.projects, 'EnableXpnHost', self.messages.ComputeProjectsEnableXpnHostRequest(project=project))
    msg = 'enable [{project}] as XPN host'.format(project=project)
    self._MakeRequestSync(request_tuple, msg)