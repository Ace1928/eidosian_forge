from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListApiProxiesResponse(_messages.Message):
    """To change this message, in the same CL add a change log in go/changing-
  api-proto-breaks-ui

  Fields:
    proxies: A GoogleCloudApigeeV1ApiProxy attribute.
  """
    proxies = _messages.MessageField('GoogleCloudApigeeV1ApiProxy', 1, repeated=True)