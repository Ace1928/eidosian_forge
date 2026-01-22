from .low_level import Message, MessageType, HeaderFields
from .wrappers import MessageGenerator, new_method_call
def UpdateActivationEnvironment(self, env):
    return new_method_call(self, 'UpdateActivationEnvironment', 'a{ss}', (env,))