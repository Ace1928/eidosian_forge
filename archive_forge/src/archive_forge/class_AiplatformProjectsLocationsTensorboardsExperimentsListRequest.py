from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsExperimentsListRequest(_messages.Message):
    """A AiplatformProjectsLocationsTensorboardsExperimentsListRequest object.

  Fields:
    filter: Lists the TensorboardExperiments that match the filter expression.
    orderBy: Field to use to sort the list.
    pageSize: The maximum number of TensorboardExperiments to return. The
      service may return fewer than this value. If unspecified, at most 50
      TensorboardExperiments are returned. The maximum value is 1000; values
      above 1000 are coerced to 1000.
    pageToken: A page token, received from a previous
      TensorboardService.ListTensorboardExperiments call. Provide this to
      retrieve the subsequent page. When paginating, all other parameters
      provided to TensorboardService.ListTensorboardExperiments must match the
      call that provided the page token.
    parent: Required. The resource name of the Tensorboard to list
      TensorboardExperiments. Format:
      `projects/{project}/locations/{location}/tensorboards/{tensorboard}`
    readMask: Mask specifying which fields to read.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
    readMask = _messages.StringField(6)