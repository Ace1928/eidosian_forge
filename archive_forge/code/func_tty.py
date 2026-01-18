from pprint import pformat
from six import iteritems
import re
@tty.setter
def tty(self, tty):
    """
        Sets the tty of this V1Container.
        Whether this container should allocate a TTY for itself, also requires
        'stdin' to be true. Default is false.

        :param tty: The tty of this V1Container.
        :type: bool
        """
    self._tty = tty