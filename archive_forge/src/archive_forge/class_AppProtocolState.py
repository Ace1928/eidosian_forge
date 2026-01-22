import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
class AppProtocolState(enum.Enum):
    STATE_INIT = 'STATE_INIT'
    STATE_CON_MADE = 'STATE_CON_MADE'
    STATE_EOF = 'STATE_EOF'
    STATE_CON_LOST = 'STATE_CON_LOST'