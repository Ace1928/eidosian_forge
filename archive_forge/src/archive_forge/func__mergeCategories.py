from fontTools.cffLib import maxStackLimit
def _mergeCategories(a, b):
    if a == '0':
        return b
    if b == '0':
        return a
    if a == b:
        return a
    return None