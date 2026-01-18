import io
import logging
import os
import sys
import threading
import time
from io import StringIO
def writeOutputBuffer(self, ostreams):
    if not self.encoding:
        ostring, self.output_buffer = (self.output_buffer, b'')
    elif self.buffering == 1:
        EOL = self.output_buffer.rfind(self.newlines or '\n') + 1
        ostring = self.output_buffer[:EOL]
        self.output_buffer = self.output_buffer[EOL:]
    else:
        ostring, self.output_buffer = (self.output_buffer, '')
    if not ostring:
        return
    for stream in ostreams:
        try:
            written = stream.write(ostring)
        except:
            written = 0
        if written and (not self.buffering):
            stream.flush()
        if written is not None and written != len(ostring):
            logger.error('Output stream (%s) closed before all output was written to it. The following was left in the output buffer:\n\t%r' % (stream, ostring[written:]))