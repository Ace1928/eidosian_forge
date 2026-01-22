from fontTools.cffLib import maxStackLimit
def _negateCategory(a):
    if a == 'h':
        return 'v'
    if a == 'v':
        return 'h'
    assert a in '0r'
    return a