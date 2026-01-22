import errno
import fcntl
import os
import subprocess
import time
from . import Connection, ConnectionException
def _xvfb_command(self):
    """ You can override this if you have some extra args for Xvfb or
        whatever. At this point, os.environ['DISPLAY'] is set to something Xvfb
        can use. """
    screen = '%sx%sx%s' % (self.width, self.height, self.depth)
    return ['Xvfb', os.environ['DISPLAY'], '-screen', '0', screen]