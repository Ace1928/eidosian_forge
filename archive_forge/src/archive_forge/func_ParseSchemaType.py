from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.pubsub.util import InvalidArgumentError
def ParseSchemaType(messages, schema_type):
    type_str = schema_type.lower()
    if type_str == 'protocol-buffer' or type_str == 'protocol_buffer':
        return messages.Schema.TypeValueValuesEnum.PROTOCOL_BUFFER
    elif type_str == 'avro':
        return messages.Schema.TypeValueValuesEnum.AVRO
    raise InvalidSchemaType()