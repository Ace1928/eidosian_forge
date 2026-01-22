from Xlib import X
from Xlib.protocol import rq, structs
def add_output_mode(self):
    return AddOutputMode(display=self.display, opcode=self.display.get_extension_major(extname), output=output, mode=mode)