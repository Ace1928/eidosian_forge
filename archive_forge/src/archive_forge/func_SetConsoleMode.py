def SetConsoleMode(handle, mode):
    success = _SetConsoleMode(handle, mode)
    if not success:
        raise ctypes.WinError()