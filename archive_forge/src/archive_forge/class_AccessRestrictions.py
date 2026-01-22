from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessRestrictions(_messages.Message):
    """Access related restrictions on the workforce pool.

  Fields:
    allowedServices: Optional. Immutable. Services allowed for web sign-in
      with the workforce pool. If not set by default there are no
      restrictions.
    disableProgrammaticSignin: Optional. Disable programmatic sign-in by
      disabling token issue via the Security Token API endpoint. See [Security
      Token Service API]
      (https://cloud.google.com/iam/docs/reference/sts/rest).
  """
    allowedServices = _messages.MessageField('ServiceConfig', 1, repeated=True)
    disableProgrammaticSignin = _messages.BooleanField(2)