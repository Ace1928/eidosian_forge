import py, sys
def _apiwarn(startversion, msg, stacklevel=2, function=None):
    if isinstance(stacklevel, str):
        frame = sys._getframe(1)
        level = 1
        found = frame.f_code.co_filename.find(stacklevel) != -1
        while frame:
            co = frame.f_code
            if co.co_filename.find(stacklevel) == -1:
                if found:
                    stacklevel = level
                    break
            else:
                found = True
            level += 1
            frame = frame.f_back
        else:
            stacklevel = 1
    msg = '%s (since version %s)' % (msg, startversion)
    warn(msg, stacklevel=stacklevel + 1, function=function)