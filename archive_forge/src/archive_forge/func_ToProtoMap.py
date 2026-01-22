from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from typing import Generator
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as messages
@staticmethod
def ToProtoMap(map_value_cls, value):
    """encoding.DictToAdditionalPropertyMessage wrapper to match ToPyDict."""
    return encoding.DictToAdditionalPropertyMessage(value, map_value_cls, sort_items=True)