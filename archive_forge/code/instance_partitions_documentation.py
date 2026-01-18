from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import list_pager
from cloudsdk.google.protobuf import timestamp_pb2
from googlecloudsdk.api_lib.spanner import response_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
Delete an instance partition.