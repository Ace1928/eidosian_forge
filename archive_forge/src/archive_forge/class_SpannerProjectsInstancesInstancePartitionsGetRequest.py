from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesInstancePartitionsGetRequest(_messages.Message):
    """A SpannerProjectsInstancesInstancePartitionsGetRequest object.

  Fields:
    name: Required. The name of the requested instance partition. Values are
      of the form `projects/{project}/instances/{instance}/instancePartitions/
      {instance_partition}`.
  """
    name = _messages.StringField(1, required=True)