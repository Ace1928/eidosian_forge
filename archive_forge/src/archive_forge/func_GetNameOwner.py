from .low_level import Message, MessageType, HeaderFields
from .wrappers import MessageGenerator, new_method_call
def GetNameOwner(self, name):
    return new_method_call(self, 'GetNameOwner', 's', (name,))