from typing import cast, Dict, Optional, Set, Tuple, Type, Union
from ._events import *
from ._util import LocalProtocolError, Sentinel
class SEND_RESPONSE(Sentinel, metaclass=Sentinel):
    pass