import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString

        Fires sentence callbacks for the current sentence.

        A callback will only fire if all of the keys it requires are present
        in the current state and at least one such field was altered in the
        current sentence.

        The callbacks will only be fired with data from L{_state}.
        