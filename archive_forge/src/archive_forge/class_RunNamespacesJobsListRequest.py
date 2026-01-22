from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunNamespacesJobsListRequest(_messages.Message):
    """A RunNamespacesJobsListRequest object.

  Fields:
    continue_: Optional. Optional encoded string to continue paging.
    fieldSelector: Optional. Not supported by Cloud Run.
    includeUninitialized: Optional. Not supported by Cloud Run.
    labelSelector: Optional. Allows to filter resources based on a label.
      Supported operations are =, !=, exists, in, and notIn.
    limit: Optional. The maximum number of records that should be returned.
    parent: Required. The namespace from which the jobs should be listed.
      Replace {namespace} with the project ID or number. It takes the form
      namespaces/{namespace}. For example: namespaces/PROJECT_ID
    resourceVersion: Optional. Not supported by Cloud Run.
    watch: Optional. Not supported by Cloud Run.
  """
    continue_ = _messages.StringField(1)
    fieldSelector = _messages.StringField(2)
    includeUninitialized = _messages.BooleanField(3)
    labelSelector = _messages.StringField(4)
    limit = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    parent = _messages.StringField(6, required=True)
    resourceVersion = _messages.StringField(7)
    watch = _messages.BooleanField(8)