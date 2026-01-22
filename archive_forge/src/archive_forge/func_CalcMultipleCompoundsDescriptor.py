from math import *
from rdkit import RDConfig
def CalcMultipleCompoundsDescriptor(composVect, argVect, atomDict, propDictList):
    """ calculates the value of the descriptor for a list of compounds

    **ARGUMENTS:**

      - composVect: a vector of vector/tuple containing the composition
         information.
         See _CalcSingleCompoundDescriptor()_ for an explanation of the elements.

      - argVect: a vector/tuple with three elements:

           1) AtomicDescriptorNames:  a list/tuple of the names of the
             atomic descriptors being used. These determine the
             meaning of $1, $2, etc. in the expression

           2) CompoundDsscriptorNames:  a list/tuple of the names of the
             compound descriptors being used. These determine the
             meaning of $a, $b, etc. in the expression

           3) Expr: a string containing the expression to be used to
             evaluate the final result.

      - atomDict:
           a dictionary of atomic descriptors.  Each atomic entry is
           another dictionary containing the individual descriptors
           and their values

      - propVectList:
         a vector of vectors of descriptors for the composition.

    **RETURNS:**

      a vector containing the values of the descriptor for each
      compound.  Any given entry will be -666 if problems were
      encountered

  """
    res = [-666] * len(composVect)
    try:
        atomVarNames = argVect[0]
        compositionVarNames = argVect[1]
        formula = argVect[2]
        formula = _SubForCompoundDescriptors(formula, compositionVarNames, 'propDict')
        formula = _SubForAtomicVars(formula, atomVarNames, 'atomDict')
        evalTarget = _SubMethodArgs(formula, knownMethods)
    except Exception:
        return res
    for i in range(len(composVect)):
        propDict = propDictList[i]
        compos = composVect[i]
        try:
            v = eval(evalTarget)
        except Exception:
            v = -666
        res[i] = v
    return res