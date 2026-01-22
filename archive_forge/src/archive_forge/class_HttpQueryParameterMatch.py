from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpQueryParameterMatch(_messages.Message):
    """HttpRouteRuleMatch criteria for a request's query parameter.

  Fields:
    exactMatch: The queryParameterMatch matches if the value of the parameter
      exactly matches the contents of exactMatch. Only one of presentMatch,
      exactMatch, or regexMatch must be set.
    name: The name of the query parameter to match. The query parameter must
      exist in the request, in the absence of which the request match fails.
    presentMatch: Specifies that the queryParameterMatch matches if the
      request contains the query parameter, irrespective of whether the
      parameter has a value or not. Only one of presentMatch, exactMatch, or
      regexMatch must be set.
    regexMatch: The queryParameterMatch matches if the value of the parameter
      matches the regular expression specified by regexMatch. For more
      information about regular expression syntax, see Syntax. Only one of
      presentMatch, exactMatch, or regexMatch must be set. Regular expressions
      can only be used when the loadBalancingScheme is set to
      INTERNAL_SELF_MANAGED.
  """
    exactMatch = _messages.StringField(1)
    name = _messages.StringField(2)
    presentMatch = _messages.BooleanField(3)
    regexMatch = _messages.StringField(4)