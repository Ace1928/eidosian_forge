from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.pubsub.util import InvalidArgumentError
def InvalidSchemaType():
    return InvalidArgumentError('The schema type must be either AVRO or PROTOCOL-BUFFER.')