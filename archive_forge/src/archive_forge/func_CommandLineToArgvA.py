from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import LocalFree
def CommandLineToArgvA(lpCmdLine):
    t_ansi = GuessStringType.t_ansi
    t_unicode = GuessStringType.t_unicode
    if isinstance(lpCmdLine, t_ansi):
        cmdline = t_unicode(lpCmdLine)
    else:
        cmdline = lpCmdLine
    return [t_ansi(x) for x in CommandLineToArgvW(cmdline)]