from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
def _GetRawKeyFunctionPosix():
    """_GetRawKeyFunction helper using Posix APIs."""
    import tty
    import termios

    def _GetRawKeyPosix():
        """Reads and returns one keypress from stdin, no echo, using Posix APIs.

    Returns:
      The key name, None for EOF, <*> for function keys, otherwise a
      character.
    """
        ansi_to_key = {'A': '<UP-ARROW>', 'B': '<DOWN-ARROW>', 'D': '<LEFT-ARROW>', 'C': '<RIGHT-ARROW>', '5': '<PAGE-UP>', '6': '<PAGE-DOWN>', 'H': '<HOME>', 'F': '<END>', 'M': '<DOWN-ARROW>', 'S': '<PAGE-UP>', 'T': '<PAGE-DOWN>'}
        sys.stdout.flush()
        fd = sys.stdin.fileno()

        def _GetKeyChar():
            return encoding.Decode(os.read(fd, 1))
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            c = _GetKeyChar()
            if c == _ANSI_CSI:
                c = _GetKeyChar()
                while True:
                    if c == _ANSI_CSI:
                        return c
                    if c.isalpha():
                        break
                    prev_c = c
                    c = _GetKeyChar()
                    if c == '~':
                        c = prev_c
                        break
                return ansi_to_key.get(c, '')
        except:
            c = None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return None if c in (_CONTROL_D, _CONTROL_Z) else c
    return _GetRawKeyPosix