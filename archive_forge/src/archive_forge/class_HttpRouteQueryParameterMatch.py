from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpRouteQueryParameterMatch(_messages.Message):
    """Specifications to match a query parameter in the request.

  Fields:
    exactMatch: The value of the query parameter must exactly match the
      contents of exact_match. Only one of exact_match, regex_match, or
      present_match must be set.
    presentMatch: Specifies that the QueryParameterMatcher matches if request
      contains query parameter, irrespective of whether the parameter has a
      value or not. Only one of exact_match, regex_match, or present_match
      must be set.
    queryParameter: The name of the query parameter to match.
    regexMatch: The value of the query parameter must match the regular
      expression specified by regex_match. For regular expression grammar,
      please see https://github.com/google/re2/wiki/Syntax Only one of
      exact_match, regex_match, or present_match must be set.
  """
    exactMatch = _messages.StringField(1)
    presentMatch = _messages.BooleanField(2)
    queryParameter = _messages.StringField(3)
    regexMatch = _messages.StringField(4)