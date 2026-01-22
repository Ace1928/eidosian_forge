from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import resources as cloud_resources
import six
def NewAPIAdapter(api_version):
    return InitAPIAdapter(api_version, APIAdapter)