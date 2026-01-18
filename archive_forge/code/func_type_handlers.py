from kombu import exceptions as kombu_exc
from taskflow import exceptions as excp
from taskflow import logging
from taskflow.utils import kombu_utils as ku
@property
def type_handlers(self):
    """Dictionary of message type -> callback to handle that message.

        The callback(s) will be activated by looking for a message
        property 'type' and locating a callback in this dictionary that maps
        to that type; if one is found it is expected to be a callback that
        accepts two positional parameters; the first being the message data
        and the second being the message object. If a callback is not found
        then the message is rejected and it will be up to the underlying
        message transport to determine what this means/implies...
        """
    return self._type_handlers