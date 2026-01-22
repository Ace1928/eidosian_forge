from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.monitoring import completers
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core.util import times
def AddUptimeProtocolFlags(parser, update=False):
    """Adds uptime check protocol settings flags to the parser."""
    uptime_protocol_group = parser.add_group(help='Uptime check protocol settings.')
    if not update:
        uptime_protocol_group.add_argument('--protocol', help='The protocol of the request, defaults to `http`.', choices=UPTIME_PROTOCOLS)
    uptime_protocol_group.add_argument('--port', help='The port on the server against which to run the check.\n        Defaults to `80` when `--protocol` is `http`.\n        Defaults to `443` when `--protocol` is `https`.\n        Required if `--protocol` is `tcp`.', type=arg_parsers.BoundedInt(lower_bound=1, upper_bound=65535))
    uptime_protocol_group.add_argument('--pings-count', help='Number of ICMP pings to send alongside the request.', type=arg_parsers.BoundedInt(lower_bound=1, upper_bound=3))
    uptime_protocol_group.add_argument('--request-method', help='The HTTP request method to use, defaults to `get`.\n        Can only be set if `--protocol` is `http` or `https`.', choices=UPTIME_REQUEST_METHODS)
    uptime_protocol_group.add_argument('--path', help='The path to the page against which to run the check, defaults to `/`.\n        Can only be set if `--protocol` is `http` or `https`.', type=str)
    uptime_protocol_group.add_argument('--username', help='The username to use when authenticating with the HTTP server.\n        Can only be set if `--protocol` is `http` or `https`.', type=str)
    uptime_protocol_group.add_argument('--password', help='The password to use when authenticating with the HTTP server.\n        Can only be set if `--protocol` is `http` or `https`.', type=str)
    uptime_protocol_group.add_argument('--mask-headers', help='Whether to encrypt the header information, defaults to `false`.\n        Can only be set if `--protocol` is `http` or `https`.', type=bool)
    if update:
        uptime_headers_group = uptime_protocol_group.add_group(help='Uptime check headers.')
        base.Argument('--update-headers', metavar='KEY=VALUE', type=arg_parsers.ArgDict(key_type=str, value_type=str), action=arg_parsers.UpdateAction, help='The list of headers to add to the uptime check. Any existing\n              headers with matching "key" are overridden by the provided\n              values.').AddToParser(uptime_headers_group)
        uptime_remove_header_group = uptime_headers_group.add_group(help='Uptime check remove headers.', mutex=True)
        uptime_remove_header_group.add_argument('--remove-headers', metavar='KEY', help='The list of header keys to remove from the uptime check.', type=arg_parsers.ArgList(str))
        uptime_remove_header_group.add_argument('--clear-headers', help='Clear all headers on the uptime check.', type=bool)
    else:
        base.Argument('--headers', metavar='KEY=VALUE', type=arg_parsers.ArgDict(key_type=str, value_type=str), action=arg_parsers.UpdateAction, help='The list of headers to send as part of the uptime check\n              request. Can only be set if `--protocol` is `http` or `https`.').AddToParser(uptime_protocol_group)
    uptime_protocol_group.add_argument('--content-type', help='The content type header to use for the check, defaults to `unspecified`.\n        Can only be set if `--protocol` is `http` or `https`.', choices=UPTIME_CONTENT_TYPES)
    uptime_protocol_group.add_argument('--custom-content-type', help='A user-provided content type header to use for the check.\n        Can only be set if `--protocol` is `http` or `https`.', type=str)
    uptime_protocol_group.add_argument('--validate-ssl', help='Whether to include SSL certificate validation as a part of the uptime check,\n        defaults to `false`.\n        Can only be set if `--protocol` is `http` or `https`.', type=bool)
    uptime_protocol_group.add_argument('--body', help='The request body associated with the HTTP POST request.\n        Can only be set if `--protocol` is `http` or `https`.', type=str)
    uptime_status_group = uptime_protocol_group.add_group(help='Uptime check status.', mutex=True)
    if update:
        uptime_status_classes_group = uptime_status_group.add_group(help='Uptime check status classes.', mutex=True)
        uptime_status_classes_group.add_argument('--set-status-classes', metavar='status-class', help='List of HTTP status classes. The uptime check will only pass if the response\n                code is contained in this list.', type=arg_parsers.ArgList(choices=UPTIME_STATUS_CLASSES))
        uptime_status_classes_group.add_argument('--add-status-classes', metavar='status-class', help='The list of HTTP status classes to add to the uptime check.', type=arg_parsers.ArgList(choices=UPTIME_STATUS_CLASSES))
        uptime_status_classes_group.add_argument('--remove-status-classes', metavar='status-class', help='The list of HTTP status classes to remove from the uptime check.', type=arg_parsers.ArgList(choices=UPTIME_STATUS_CLASSES))
        uptime_status_classes_group.add_argument('--clear-status-classes', help='Clear all HTTP status classes on the uptime check. Setting this\n            flag is the same as selecting only the `2xx` status class.', type=bool)
        uptime_status_codes_group = uptime_status_group.add_group(help='Uptime check status codes.', mutex=True)
        uptime_status_codes_group.add_argument('--set-status-codes', metavar='status-code', help='List of HTTP status codes. The uptime check will only pass if the response\n                code is present in this list.', type=arg_parsers.ArgList(int))
        uptime_status_codes_group.add_argument('--add-status-codes', metavar='status-code', help='The list of HTTP status codes to add to the uptime check.', type=arg_parsers.ArgList(int))
        uptime_status_codes_group.add_argument('--remove-status-codes', metavar='status-code', help='The list of HTTP status codes to remove from the uptime check.', type=arg_parsers.ArgList(int))
        uptime_status_codes_group.add_argument('--clear-status-codes', help='Clear all HTTP status codes on the uptime check. Setting this\n            flag is the same as selecting only the `2xx` status class.', type=bool)
    else:
        uptime_status_group.add_argument('--status-classes', metavar='status-class', help='List of HTTP status classes. The uptime check only passes when the response\n              code is contained in this list. Defaults to `2xx`.\n              Can only be set if `--protocol` is `http` or `https`.', type=arg_parsers.ArgList(choices=UPTIME_STATUS_CLASSES))
        uptime_status_group.add_argument('--status-codes', metavar='status-code', help='List of HTTP Status Codes. The uptime check will only pass if the response code\n              is present in this list.\n              Can only be set if `--protocol` is `http` or `https`.', type=arg_parsers.ArgList(int))