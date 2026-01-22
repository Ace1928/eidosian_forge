import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
def _switch_this(self, this, broker):
    """Switch the underlying C-library pointer for this object.

        As all useful state for a Dict is stored by the underlying C-library
        pointer, it is very convenient to allow this to be switched at
        run-time.  Pass a new dict data object into this method to affect
        the necessary changes.  The creating Broker object (at the Python
        level) must also be provided.

        This should *never* *ever* be used by application code.  It's
        a convenience for developers only, replacing the clunkier <data>
        parameter to __init__ from earlier versions.
        """
    Dict._free(self)
    self._this = this
    self._broker = broker
    desc = self.__describe(check_this=False)
    self.tag = desc[0]
    self.provider = ProviderDesc(*desc[1:])