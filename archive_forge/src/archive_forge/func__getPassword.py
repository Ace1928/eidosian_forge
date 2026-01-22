import contextlib
import getpass
import io
import os
import sys
from base64 import decodebytes
from twisted.conch.client import agent
from twisted.conch.client.knownhosts import ConsoleUI, KnownHostsFile
from twisted.conch.error import ConchError
from twisted.conch.ssh import common, keys, userauth
from twisted.internet import defer, protocol, reactor
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
def _getPassword(self, prompt):
    """
        Prompt for a password using L{getpass.getpass}.

        @param prompt: Written on tty to ask for the input.
        @type prompt: L{str}
        @return: The input.
        @rtype: L{str}
        """
    with self._replaceStdoutStdin():
        try:
            p = getpass.getpass(prompt)
            return p
        except (KeyboardInterrupt, OSError):
            print()
            raise ConchError('PEBKAC')