from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import re
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import http_encoding
import six
def ParseExistingMessageIntoMessage(message, existing_message, method):
    """Sets fields in message based on an existing message.

  This function is used for get-modify-update pattern. The request type of
  update requests would be either the same as the response type of get requests
  or one field inside the request would be the same as the get response.

  For example:
  1) update.request_type_name = ServiceAccount
     get.response_type_name = ServiceAccount
  2) update.request_type_name = updateInstanceRequest
     updateInstanceRequest.instance = Instance
     get.response_type_name = Instance

  If the existing message has the same type as the message to be sent for the
  request, then return the existing message instead. If they are different, find
  the field in the message which has the same type as existing_message, then
  assign exsiting message to that field.

  Args:
    message: the apitools message to construct a new request.
    existing_message: the exsting apitools message returned from server.
    method: APIMethod, the method to generate request for.

  Returns:
    A modified apitools message to be send to the method.
  """
    if type(existing_message) == type(message):
        return existing_message
    field_path = method.request_field
    field = message.field_by_name(method.request_field)
    if field.message_type != type(existing_message):
        nested_message = field.message_type()
        for nested_field in nested_message.all_fields():
            try:
                if nested_field.message_type == type(existing_message):
                    field_path += '.' + nested_field.name
                    break
            except AttributeError:
                pass
    SetFieldInMessage(message, field_path, existing_message)
    return message