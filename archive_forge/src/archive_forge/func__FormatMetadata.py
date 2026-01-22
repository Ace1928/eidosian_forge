from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import io
from apitools.base.py import extra_types
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.core.resource import resource_printer
def _FormatMetadata(self, resp):
    resp_message = extra_types.encoding.MessageToPyValue(resp)
    if 'queryResult' in resp_message:
        if 'rows' in resp_message['queryResult']:
            del resp_message['queryResult']['rows']
        if 'schema' in resp_message['queryResult']:
            del resp_message['queryResult']['schema']
        if not resp_message['queryResult']:
            del resp_message['queryResult']
    string_buf = io.StringIO()
    resource_printer.Print(resp_message, 'yaml', out=string_buf)
    return string_buf.getvalue()