from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceDirectoryConfig(_messages.Message):
    """ServiceDirectoryConfig represents Service Directory configuration for a
  SCM host connection.

  Fields:
    service: The Service Directory service name. Format: projects/{project}/lo
      cations/{location}/namespaces/{namespace}/services/{service}.
  """
    service = _messages.StringField(1)