from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotifierSecretRef(_messages.Message):
    """NotifierSecretRef contains the reference to a secret stored in the
  corresponding NotifierSpec.

  Fields:
    secretRef: The value of `secret_ref` should be a `name` that is registered
      in a `Secret` in the `secrets` list of the `Spec`.
  """
    secretRef = _messages.StringField(1)