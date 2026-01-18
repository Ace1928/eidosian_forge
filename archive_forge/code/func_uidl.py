import errno
import re
import socket
import sys
def uidl(self, which=None):
    """Return message digest (unique id) list.

        If 'which', result contains unique id for that message
        in the form 'response mesgnum uid', otherwise result is
        the list ['response', ['mesgnum uid', ...], octets]
        """
    if which is not None:
        return self._shortcmd('UIDL %s' % which)
    return self._longcmd('UIDL')