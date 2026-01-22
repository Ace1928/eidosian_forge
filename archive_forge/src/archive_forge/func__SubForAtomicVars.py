from math import *
from rdkit import RDConfig
def _SubForAtomicVars(cExpr, varList, dictName):
    """ replace atomic variables with the appropriate dictionary lookup

   *Not intended for client use*

  """
    for i in range(len(varList)):
        cExpr = cExpr.replace('$%d' % (i + 1), '%s["DEADBEEF"]["%s"]' % (dictName, varList[i]))
    return cExpr