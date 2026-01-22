from typing import Optional
from twisted.python.deprecate import getWarningMethod
from twisted.python.failure import Failure
from twisted.python.log import err
from twisted.python.reflect import qual

        Call processEnded on protocol after final cleanup.
        