from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TestFailure(_messages.Message):
    """A TestFailure object.

  Fields:
    actualOutputUrl: The actual output URL evaluated by a load balancer
      containing the scheme, host, path and query parameters.
    actualRedirectResponseCode: Actual HTTP status code for rule with
      `urlRedirect` calculated by load balancer
    actualService: BackendService or BackendBucket returned by load balancer.
    expectedOutputUrl: The expected output URL evaluated by a load balancer
      containing the scheme, host, path and query parameters.
    expectedRedirectResponseCode: Expected HTTP status code for rule with
      `urlRedirect` calculated by load balancer
    expectedService: Expected BackendService or BackendBucket resource the
      given URL should be mapped to.
    headers: HTTP headers of the request.
    host: Host portion of the URL.
    path: Path portion including query parameters in the URL.
  """
    actualOutputUrl = _messages.StringField(1)
    actualRedirectResponseCode = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    actualService = _messages.StringField(3)
    expectedOutputUrl = _messages.StringField(4)
    expectedRedirectResponseCode = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    expectedService = _messages.StringField(6)
    headers = _messages.MessageField('UrlMapTestHeader', 7, repeated=True)
    host = _messages.StringField(8)
    path = _messages.StringField(9)