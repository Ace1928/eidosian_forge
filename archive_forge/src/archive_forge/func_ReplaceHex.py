import re
def ReplaceHex(m):
    if len(m.group(1)) & 1:
        return m.group(1) + 'x0' + m.group(2)
    return m.group(0)