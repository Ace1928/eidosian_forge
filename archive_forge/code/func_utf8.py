import errno
import re
import socket
import sys
def utf8(self):
    """Try to enter UTF-8 mode (see RFC 6856). Returns server response.
        """
    return self._shortcmd('UTF8')