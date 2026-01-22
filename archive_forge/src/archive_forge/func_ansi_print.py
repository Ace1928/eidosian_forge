import sys, os, unicodedata
import py
from py.builtin import text, bytes
def ansi_print(text, esc, file=None, newline=True, flush=False):
    if file is None:
        file = sys.stderr
    text = text.rstrip()
    if esc and (not isinstance(esc, tuple)):
        esc = (esc,)
    if esc and sys.platform != 'win32' and file.isatty():
        text = ''.join(['\x1b[%sm' % cod for cod in esc]) + text + '\x1b[0m'
    if newline:
        text += '\n'
    if esc and win32_and_ctypes and file.isatty():
        if 1 in esc:
            bold = True
            esc = tuple([x for x in esc if x != 1])
        else:
            bold = False
        esctable = {(): FOREGROUND_WHITE, (31,): FOREGROUND_RED, (32,): FOREGROUND_GREEN, (33,): FOREGROUND_GREEN | FOREGROUND_RED, (34,): FOREGROUND_BLUE, (35,): FOREGROUND_BLUE | FOREGROUND_RED, (36,): FOREGROUND_BLUE | FOREGROUND_GREEN, (37,): FOREGROUND_WHITE, (39,): FOREGROUND_WHITE}
        attr = esctable.get(esc, FOREGROUND_WHITE)
        if bold:
            attr |= FOREGROUND_INTENSITY
        STD_OUTPUT_HANDLE = -11
        STD_ERROR_HANDLE = -12
        if file is sys.stderr:
            handle = GetStdHandle(STD_ERROR_HANDLE)
        else:
            handle = GetStdHandle(STD_OUTPUT_HANDLE)
        oldcolors = GetConsoleInfo(handle).wAttributes
        attr |= oldcolors & 240
        SetConsoleTextAttribute(handle, attr)
        while len(text) > 32768:
            file.write(text[:32768])
            text = text[32768:]
        if text:
            file.write(text)
        SetConsoleTextAttribute(handle, oldcolors)
    else:
        file.write(text)
    if flush:
        file.flush()