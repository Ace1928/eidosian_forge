import os
import shlex
import subprocess
class WindowsParser:
    """
    The parsing behavior used by `subprocess.call("string")` on Windows, which
    matches the Microsoft C/C++ runtime.

    Note that this is _not_ the behavior of cmd.
    """

    @staticmethod
    def join(argv):
        return subprocess.list2cmdline(argv)

    @staticmethod
    def split(cmd):
        import ctypes
        try:
            ctypes.windll
        except AttributeError:
            raise NotImplementedError
        if not cmd:
            return []
        cmd = 'dummy ' + cmd
        CommandLineToArgvW = ctypes.windll.shell32.CommandLineToArgvW
        CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
        CommandLineToArgvW.argtypes = (ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int))
        nargs = ctypes.c_int()
        lpargs = CommandLineToArgvW(cmd, ctypes.byref(nargs))
        args = [lpargs[i] for i in range(nargs.value)]
        assert not ctypes.windll.kernel32.LocalFree(lpargs)
        assert args[0] == 'dummy'
        return args[1:]