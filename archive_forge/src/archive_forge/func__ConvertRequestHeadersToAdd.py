from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
def _ConvertRequestHeadersToAdd(self, request_headers_to_add):
    """Converts a request-headers-to-add string list into an HttpHeaderAction.

    Args:
      request_headers_to_add: A dict of headers to add to requests that match
        this rule. Leading whitespace in each header name and value is stripped.

    Returns:
      An HttpHeaderAction object with a populated request_headers_to_add field.
    """
    header_action = self._messages.SecurityPolicyRuleHttpHeaderAction()
    for hdr_name, hdr_val in request_headers_to_add.items():
        header_to_add = self._messages.SecurityPolicyRuleHttpHeaderActionHttpHeaderOption()
        header_to_add.headerName = hdr_name.strip()
        header_to_add.headerValue = hdr_val.lstrip()
        header_action.requestHeadersToAdds.append(header_to_add)
    return header_action