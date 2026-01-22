from .messages import Message
from .midifiles import MetaMessage, UnknownMetaMessage
class FrozenMetaMessage(Frozen, MetaMessage):
    pass