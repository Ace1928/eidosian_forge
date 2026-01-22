from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
def GetUriFunc(self):

    def _GetUri(search_result):
        return ''.join([p.value.string_value for p in search_result.resource.additionalProperties if p.key == 'selfLink'])
    return _GetUri