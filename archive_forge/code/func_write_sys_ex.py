from midi devices.  It can also list midi devices on the system.
import math
import atexit
import pygame
import pygame.locals
import pygame.pypm as _pypm
def write_sys_ex(self, when, msg):
    """writes a timestamped system-exclusive midi message.
        Output.write_sys_ex(when, msg)

        msg - can be a *list* or a *string*
        when - a timestamp in milliseconds
        example:
          (assuming o is an onput MIDI stream)
            o.write_sys_ex(0,'\\xF0\\x7D\\x10\\x11\\x12\\x13\\xF7')
          is equivalent to
            o.write_sys_ex(pygame.midi.time(),
                           [0xF0,0x7D,0x10,0x11,0x12,0x13,0xF7])
        """
    _check_init()
    self._check_open()
    self._output.WriteSysEx(when, msg)