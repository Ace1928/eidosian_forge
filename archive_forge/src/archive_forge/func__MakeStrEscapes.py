import re
def _MakeStrEscapes():
    ret = {}
    for i in range(0, 128):
        if not _AsciiIsPrint(i):
            ret[i] = '\\%03o' % i
    ret[ord('\t')] = '\\t'
    ret[ord('\n')] = '\\n'
    ret[ord('\r')] = '\\r'
    ret[ord('"')] = '\\"'
    ret[ord("'")] = "\\'"
    ret[ord('\\')] = '\\\\'
    return ret