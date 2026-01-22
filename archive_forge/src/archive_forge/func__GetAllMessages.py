from cloudsdk.google.protobuf.internal import api_implementation
from cloudsdk.google.protobuf import descriptor_pool
from cloudsdk.google.protobuf import message_factory
def _GetAllMessages(desc):
    """Walk a message Descriptor and recursively yields all message names."""
    yield desc
    for msg_desc in desc.nested_types:
        for nested_desc in _GetAllMessages(msg_desc):
            yield nested_desc