import os
import struct
from codecs import getincrementaldecoder, IncrementalDecoder
from enum import IntEnum
from typing import Generator, List, NamedTuple, Optional, Tuple, TYPE_CHECKING, Union
def _parse_more_gen(self) -> Generator[Optional[Frame], None, None]:
    self.extensions = [ext for ext in self.extensions if ext.enabled()]
    closed = False
    while not closed:
        frame = self._frame_decoder.process_buffer()
        if frame is not None:
            if not frame.opcode.iscontrol():
                frame = self._message_decoder.process_frame(frame)
            elif frame.opcode == Opcode.CLOSE:
                frame = self._process_close(frame)
                closed = True
        yield frame