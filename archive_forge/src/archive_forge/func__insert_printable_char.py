import curses
import curses.ascii
def _insert_printable_char(self, ch):
    self._update_max_yx()
    y, x = self.win.getyx()
    backyx = None
    while y < self.maxy or x < self.maxx:
        if self.insert_mode:
            oldch = self.win.inch()
        try:
            self.win.addch(ch)
        except curses.error:
            pass
        if not self.insert_mode or not curses.ascii.isprint(oldch):
            break
        ch = oldch
        y, x = self.win.getyx()
        if backyx is None:
            backyx = (y, x)
    if backyx is not None:
        self.win.move(*backyx)