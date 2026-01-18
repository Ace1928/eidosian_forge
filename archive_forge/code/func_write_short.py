from midi devices.  It can also list midi devices on the system.
import math
import atexit
import pygame
import pygame.locals
import pygame.pypm as _pypm
def write_short(self, status, data1=0, data2=0):
    """write_short(status <, data1><, data2>)
        Output.write_short(status)
        Output.write_short(status, data1 = 0, data2 = 0)

        output MIDI information of 3 bytes or less.
        data fields are optional
        status byte could be:
             0xc0 = program change
             0x90 = note on
             etc.
             data bytes are optional and assumed 0 if omitted
        example: note 65 on with velocity 100
             write_short(0x90,65,100)
        """
    _check_init()
    self._check_open()
    self._output.WriteShort(status, data1, data2)