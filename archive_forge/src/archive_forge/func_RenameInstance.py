from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as apilib_exceptions
from googlecloudsdk.command_lib.bms import util
import six
def RenameInstance(self, instance_resource, new_name):
    """Rename an existing instance resource."""
    rename_instance_request = self.messages.RenameInstanceRequest(newInstanceId=new_name)
    request = self.messages.BaremetalsolutionProjectsLocationsInstancesRenameRequest(name=instance_resource.RelativeName(), renameInstanceRequest=rename_instance_request)
    return self.instances_service.Rename(request)