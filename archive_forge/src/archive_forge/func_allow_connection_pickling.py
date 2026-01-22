import os
import sys
import threading
from . import process
from . import reduction
def allow_connection_pickling(self):
    """Install support for sending connections and sockets
        between processes
        """
    from . import connection