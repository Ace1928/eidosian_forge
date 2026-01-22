import os
import tempfile
import time
import re
def RaiseWindowNamed(nameRe):
    cb = lambda x, y: y.append(x)
    wins = []
    win32gui.EnumWindows(cb, wins)
    tgtWin = -1
    for win in wins:
        txt = win32gui.GetWindowText(win)
        if nameRe.match(txt):
            tgtWin = win
            break
    if tgtWin >= 0:
        win32gui.ShowWindow(tgtWin, 1)
        win32gui.BringWindowToTop(tgtWin)