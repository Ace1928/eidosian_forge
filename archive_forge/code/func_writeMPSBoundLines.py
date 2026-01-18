import re
from . import constants as const
def writeMPSBoundLines(name, variable, mip):
    if variable.lowBound is not None and variable.lowBound == variable.upBound:
        return [' FX BND       %-8s  % .12e\n' % (name, variable.lowBound)]
    elif variable.lowBound == 0 and variable.upBound == 1 and mip and (variable.cat == const.LpInteger):
        return [' BV BND       %-8s\n' % name]
    bound_lines = []
    if variable.lowBound is not None:
        if variable.lowBound != 0 or (mip and variable.cat == const.LpInteger and (variable.upBound is None)):
            bound_lines.append(' LO BND       %-8s  % .12e\n' % (name, variable.lowBound))
    elif variable.upBound is not None:
        bound_lines.append(' MI BND       %-8s\n' % name)
    else:
        bound_lines.append(' FR BND       %-8s\n' % name)
    if variable.upBound is not None:
        bound_lines.append(' UP BND       %-8s  % .12e\n' % (name, variable.upBound))
    return bound_lines