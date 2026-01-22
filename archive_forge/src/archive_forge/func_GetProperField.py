from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from cloudsdk.google.protobuf import timestamp_pb2
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.privateca import text_utils
def GetProperField(field_name, is_log_entry):
    """Retrieve the proper atrribute from LogEntry depending if it is in MessageModule or GapiClient format."""
    if not is_log_entry:
        return field_name
    return text_utils.SnakeCaseToCamelCase(field_name)