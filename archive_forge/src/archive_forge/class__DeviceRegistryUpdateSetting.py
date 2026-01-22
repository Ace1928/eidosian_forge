from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
class _DeviceRegistryUpdateSetting(object):
    """Small value class holding data for updating a device registry."""

    def __init__(self, field_name, update_mask, value):
        self.field_name = field_name
        self.update_mask = update_mask
        self.value = value