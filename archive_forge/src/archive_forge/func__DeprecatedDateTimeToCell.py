from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import property_selector
import six
import six.moves.http_client
def _DeprecatedDateTimeToCell(zone_or_region):
    """Returns the turndown timestamp of a deprecated machine or ''."""
    deprecated = zone_or_region.get('deprecated', '')
    if deprecated:
        return deprecated.get('deleted')
    else:
        return ''