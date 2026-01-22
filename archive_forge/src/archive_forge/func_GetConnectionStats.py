from .low_level import Message, MessageType, HeaderFields
from .wrappers import MessageGenerator, new_method_call
def GetConnectionStats(self, arg0):
    return new_method_call(self, 'GetConnectionStats', 's', (arg0,))