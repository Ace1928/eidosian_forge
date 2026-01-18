import threading
import rtmidi
from .. import ports
from ..messages import Message
from ._parser_queue import ParserQueue
from .rtmidi_utils import expand_alsa_port_name
Send a message on the port.