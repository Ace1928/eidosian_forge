from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
@staticmethod
def _EnableComputeApi():
    return properties.VALUES.compute.use_new_list_usable_subnets_api.GetBool()