import os
def _closure(hWnd, wndProc):
    oldAddr = func(hWnd, GWL_WNDPROC, cast(wndProc, c_void_p).value)
    return cast(c_void_p(oldAddr), WNDPROC)