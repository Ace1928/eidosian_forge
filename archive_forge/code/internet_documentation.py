from random import random as _goodEnoughRandom
from typing import List
from automat import MethodicalMachine
from twisted.application import service
from twisted.internet import task
from twisted.internet.defer import (
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure

        Stop attempting to reconnect and close any existing connections.

        @return: a L{Deferred} that fires when all outstanding connections are
            closed and all in-progress connection attempts halted.
        