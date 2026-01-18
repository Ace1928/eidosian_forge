import sys
import os
from os import path
from contextlib import contextmanager
@toolkit.setter
def toolkit(self, toolkit):
    """
        Property setter for the GUI toolkit.  The toolkit can be set more than
        once, but only if it is the same one each time.  An application that is
        written for a particular toolkit can explicitly set it before any other
        module that gets the value is imported.

        """
    if self._toolkit and self._toolkit != toolkit:
        raise ValueError('cannot set toolkit to %s because it has already been set to %s' % (toolkit, self._toolkit))
    self._toolkit = toolkit