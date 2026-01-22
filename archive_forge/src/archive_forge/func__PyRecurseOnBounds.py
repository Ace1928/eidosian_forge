import numpy
from rdkit.ML.InfoTheory import entropy
def _PyRecurseOnBounds(vals, cuts, which, starts, results, nPossibleRes, varTable=None):
    """ Primarily intended for internal use

     Recursively finds the best quantization boundaries

     **Arguments**

       - vals: a 1D Numeric array with the values of the variables,
         this should be sorted

       - cuts: a list with the indices of the quantization bounds
         (indices are into _starts_ )

       - which: an integer indicating which bound is being adjusted here
         (and index into _cuts_ )

       - starts: a list of potential starting points for quantization bounds

       - results: a 1D Numeric array of integer result codes

       - nPossibleRes: an integer with the number of possible result codes

     **Returns**

       - a 2-tuple containing:

         1) the best information gain found so far

         2) a list of the quantization bound indices ( _cuts_ for the best case)

     **Notes**

      - this is not even remotely efficient, which is why a C replacement
        was written

    """
    nBounds = len(cuts)
    maxGain = -1000000.0
    bestCuts = None
    highestCutHere = len(starts) - nBounds + which
    if varTable is None:
        varTable = _GenVarTable(vals, cuts, starts, results, nPossibleRes)
    while cuts[which] <= highestCutHere:
        varTable = _GenVarTable(vals, cuts, starts, results, nPossibleRes)
        gainHere = entropy.InfoGain(varTable)
        if gainHere > maxGain:
            maxGain = gainHere
            bestCuts = cuts[:]
        if which < nBounds - 1:
            gainHere, cutsHere = _RecurseOnBounds(vals, cuts[:], which + 1, starts, results, nPossibleRes, varTable=varTable)
            if gainHere > maxGain:
                maxGain = gainHere
                bestCuts = cutsHere
        cuts[which] += 1
        for i in range(which + 1, nBounds):
            if cuts[i] == cuts[i - 1]:
                cuts[i] += 1
    return (maxGain, bestCuts)