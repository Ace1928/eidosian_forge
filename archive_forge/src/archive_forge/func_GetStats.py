from .low_level import Message, MessageType, HeaderFields
from .wrappers import MessageGenerator, new_method_call
def GetStats(self):
    return new_method_call(self, 'GetStats')