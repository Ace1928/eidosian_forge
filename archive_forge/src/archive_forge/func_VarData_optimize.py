from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
def VarData_optimize(self):
    return VarData_calculateNumShorts(self, optimize=True)