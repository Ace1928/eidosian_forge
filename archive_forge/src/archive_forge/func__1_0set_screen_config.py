from Xlib import X
from Xlib.protocol import rq, structs
def _1_0set_screen_config(self, size_id, rotation, config_timestamp, timestamp=X.CurrentTime):
    """Sets the screen to the specified size and rotation.

    """
    return _1_0SetScreenConfig(display=self.display, opcode=self.display.get_extension_major(extname), drawable=self, timestamp=timestamp, config_timestamp=config_timestamp, size_id=size_id, rotation=rotation)