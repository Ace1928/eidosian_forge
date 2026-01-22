from .low_level import Message, MessageType, HeaderFields
from .wrappers import MessageGenerator, new_method_call
def GetConnectionCredentials(self, name):
    return new_method_call(self, 'GetConnectionCredentials', 's', (name,))