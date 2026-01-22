import curses
import curses.ascii
def _end_of_line(self, y):
    """Go to the location of the first blank on the given line,
        returning the index of the last non-blank character."""
    self._update_max_yx()
    last = self.maxx
    while True:
        if curses.ascii.ascii(self.win.inch(y, last)) != curses.ascii.SP:
            last = min(self.maxx, last + 1)
            break
        elif last == 0:
            break
        last = last - 1
    return last