from .low_level import Message, MessageType, HeaderFields
from .wrappers import MessageGenerator, new_method_call
def RequestName(self, name, flags=0):
    return new_method_call(self, 'RequestName', 'su', (name, flags))