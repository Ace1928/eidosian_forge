from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayApiConfigOpenApiDocument(_messages.Message):
    """An OpenAPI Specification Document describing an API.

  Fields:
    document: The OpenAPI Specification document file.
  """
    document = _messages.MessageField('ApigatewayApiConfigFile', 1)