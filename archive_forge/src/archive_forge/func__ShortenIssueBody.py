from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import sys
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr_os
import six
from six.moves import range
from six.moves import urllib
def _ShortenIssueBody(comment, url_encoded_length):
    """Shortens the comment to be at most the given length (URL-encoded).

  Does one of two things:

  (1) If the whole stacktrace and everything before it fits within the
      URL-encoded max length, truncates the remainder of the comment (which
      should include e.g. the output of `gcloud info`.
  (2) Otherwise, chop out the middle of the stacktrace until it fits. (See
      _ShortenStacktrace docstring for an example).
  (3) If the stacktrace cannot be shortened to fit in (2), then revert to (1).
      That is, truncate the comment.

  Args:
    comment: CommentHolder, an object containing the formatted comment for
        inclusion before shortening, and its constituent components
    url_encoded_length: the max length of the comment after shortening (when
        comment is URL-encoded).

  Returns:
    (str, str): the shortened comment and a message containing the parts of the
    comment that were omitted by the shortening process.
  """
    critical_info, middle, optional_info = comment.body.partition('Installation information:\n')
    optional_info = middle + optional_info
    max_str_len = url_encoded_length - _UrlEncodeLen(TRUNCATED_INFO_MESSAGE + '\n')
    truncated_issue_body, remaining = _UrlTruncateLines(comment.body, max_str_len)
    if _UrlEncodeLen(critical_info) <= max_str_len:
        return (truncated_issue_body, remaining)
    else:
        non_stacktrace_encoded_len = _UrlEncodeLen(comment.pre_stacktrace + 'Trace:\n' + comment.exception + '\n' + TRUNCATED_INFO_MESSAGE)
        max_stacktrace_len = url_encoded_length - non_stacktrace_encoded_len
        shortened_stacktrace = _ShortenStacktrace(comment.stacktrace, max_stacktrace_len)
        critical_info_with_shortened_stacktrace = comment.pre_stacktrace + 'Trace:\n' + shortened_stacktrace + comment.exception + '\n' + TRUNCATED_INFO_MESSAGE
        optional_info_with_full_stacktrace = 'Full stack trace (formatted):\n' + comment.stacktrace + comment.exception + '\n\n' + optional_info
        if _UrlEncodeLen(critical_info_with_shortened_stacktrace) <= max_str_len:
            return (critical_info_with_shortened_stacktrace, optional_info_with_full_stacktrace)
        else:
            return (truncated_issue_body, optional_info_with_full_stacktrace)