from .hyperboloid_utilities import *
import time
import sys
import tempfile
import png
def tkButton1(self, event):
    for last, release in self.key_to_last_accounted_and_release_time.values():
        if last or release:
            return
    self.configure(cursor=_default_move_cursor)
    self.mouse_pos_when_pressed = (event.x, event.y)
    self.view_state_when_pressed = self.view_state
    self.mouse_mode = 'move'