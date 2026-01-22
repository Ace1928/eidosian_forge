from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OwnerReference(_messages.Message):
    """This is not supported or used by Cloud Run.

  Fields:
    apiVersion: This is not supported or used by Cloud Run.
    blockOwnerDeletion: This is not supported or used by Cloud Run.
    controller: This is not supported or used by Cloud Run.
    kind: This is not supported or used by Cloud Run.
    name: This is not supported or used by Cloud Run.
    uid: This is not supported or used by Cloud Run.
  """
    apiVersion = _messages.StringField(1)
    blockOwnerDeletion = _messages.BooleanField(2)
    controller = _messages.BooleanField(3)
    kind = _messages.StringField(4)
    name = _messages.StringField(5)
    uid = _messages.StringField(6)