from . import screen
from . import FSM
import string
def write_ch(self, ch):
    """This puts a character at the current cursor position. The cursor
        position is moved forward with wrap-around, but no scrolling is done if
        the cursor hits the lower-right corner of the screen. """
    if isinstance(ch, bytes):
        ch = self._decode(ch)
    ch = ch[0]
    if ch == u'\r':
        self.cr()
        return
    if ch == u'\n':
        self.crlf()
        return
    if ch == chr(screen.BS):
        self.cursor_back()
        return
    self.put_abs(self.cur_r, self.cur_c, ch)
    old_r = self.cur_r
    old_c = self.cur_c
    self.cursor_forward()
    if old_c == self.cur_c:
        self.cursor_down()
        if old_r != self.cur_r:
            self.cursor_home(self.cur_r, 1)
        else:
            self.scroll_up()
            self.cursor_home(self.cur_r, 1)
            self.erase_line()