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
def RenameVolume(self, volume_resource, new_name):
    """Rename an existing volume resource."""
    rename_volume_request = self.messages.RenameVolumeRequest(newVolumeId=new_name)
    request = self.messages.BaremetalsolutionProjectsLocationsVolumesRenameRequest(name=volume_resource.RelativeName(), renameVolumeRequest=rename_volume_request)
    return self.volumes_service.Rename(request)