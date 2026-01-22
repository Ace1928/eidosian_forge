from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceusageServicesGenerateIdentityRequest(_messages.Message):
    """A ServiceusageServicesGenerateIdentityRequest object.

  Fields:
    parent: Name of the consumer and service to generate an identity for.  The
      `GenerateServiceIdentity` methods currently only support projects.  An
      example name would be: `projects/123/services/example.googleapis.com`
      where `123` is the project number.
  """
    parent = _messages.StringField(1, required=True)