from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MatchRule(_messages.Message):
    """A collection of match conditions (such as query, header, or URI) for a
  request.

  Fields:
    fullPathMatch: Optional. To satisfy the `MatchRule` condition, the path of
      the request must exactly match the value specified in `full_path_match`
      after removing any query parameters and anchors that might be part of
      the original URL. `full_path_match` must begin with a `/`. The value
      must be between 1 and 1024 characters, (inclusive). One of prefix_match,
      `full_path_match`, or path_template_match must be specified.
    headerMatches: Optional. A list of HeaderMatch criteria, all of which must
      match corresponding headers in the request. You can specify up to three
      headers to match on.
    ignoreCase: Optional. Specifies that prefix_match and full_path_match
      matches are not case sensitive. The default value is `false`, which
      means that matches are case sensitive.
    pathTemplateMatch: Optional. To satisfy the `MatchRule` condition, the
      path of the request must match the wildcard pattern specified in
      `path_template_match` after removing any query parameters and anchors
      that might be part of the original URL. `path_template_match` must be
      between 1 and 255 characters (inclusive). The pattern specified by
      `path_template_match` can have at most 10 wildcard operators and 10
      variable captures. One of prefix_match, full_path_match, or
      `path_template_match` must be specified.
    prefixMatch: Optional. To satisfy the `MatchRule` condition, the request's
      path must begin with the specified `prefix_match`. `prefix_match` must
      begin with a `/`. The value must be between 1 and 1024 characters
      (inclusive). One of `prefix_match`, full_path_match, or
      path_template_match must be specified.
    queryParameterMatches: Optional. A list of QueryParameterMatcher criteria,
      all of which must match corresponding query parameters in the request.
      You can specify up to five query parameters to match on.
  """
    fullPathMatch = _messages.StringField(1)
    headerMatches = _messages.MessageField('HeaderMatch', 2, repeated=True)
    ignoreCase = _messages.BooleanField(3)
    pathTemplateMatch = _messages.StringField(4)
    prefixMatch = _messages.StringField(5)
    queryParameterMatches = _messages.MessageField('QueryParameterMatcher', 6, repeated=True)