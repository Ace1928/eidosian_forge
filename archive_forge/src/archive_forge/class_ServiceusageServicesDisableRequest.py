from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceusageServicesDisableRequest(_messages.Message):
    """A ServiceusageServicesDisableRequest object.

  Fields:
    disableServiceRequest: A DisableServiceRequest resource to be passed as
      the request body.
    name: Name of the consumer and service to disable the service on. The
      enable and disable methods currently only support projects. An example
      name would be: `projects/123/services/serviceusage.googleapis.com` where
      `123` is the project number.
  """
    disableServiceRequest = _messages.MessageField('DisableServiceRequest', 1)
    name = _messages.StringField(2, required=True)