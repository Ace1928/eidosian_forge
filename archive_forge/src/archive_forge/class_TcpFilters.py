from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TcpFilters(_messages.Message):
    """TCP filters configuration for the endpoint.

  Fields:
    tcpFilters: Required. The list of URLs to TcpFilter resources enabled for
      xDS clients using this configuration. Only filters that handle inbound
      connection and stream events must be specified. These filters work in
      conjunction with a default set of TCP filters that may already be
      configured by Traffic Director. Traffic Director will determine the
      final location of these filters within xDS configuration based on the
      name of the TCP filter. If Traffic Director positions multiple filters
      at the same location, those filters will be in the same order as
      specified in this list.
  """
    tcpFilters = _messages.StringField(1, repeated=True)