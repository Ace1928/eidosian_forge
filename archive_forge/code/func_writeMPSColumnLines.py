import re
from . import constants as const
def writeMPSColumnLines(cv, variable, mip, name, cobj, objName):
    columns_lines = []
    if mip and variable.cat == const.LpInteger:
        columns_lines.append("    MARK      'MARKER'                 'INTORG'\n")
    _tmp = ['    %-8s  %-8s  % .12e\n' % (name, k, v) for k, v in cv.items()]
    columns_lines.extend(_tmp)
    if variable in cobj:
        columns_lines.append('    %-8s  %-8s  % .12e\n' % (name, objName, cobj[variable]))
    if mip and variable.cat == const.LpInteger:
        columns_lines.append("    MARK      'MARKER'                 'INTEND'\n")
    return columns_lines