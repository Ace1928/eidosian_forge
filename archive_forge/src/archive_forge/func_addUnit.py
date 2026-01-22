import unicodedata
def addUnit(prefix, val):
    g = globals()
    for u in UNITS:
        g[prefix + u] = val
        allUnits[prefix + u] = val