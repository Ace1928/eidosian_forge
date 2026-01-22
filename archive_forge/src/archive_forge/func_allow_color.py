import os
import sys
def allow_color():
    if os.name != 'posix':
        return False
    if not sys.stdout.isatty():
        return False
    try:
        import curses
        curses.setupterm()
        return curses.tigetnum('colors') > 2
    except curses.error:
        return False