from warnings import warn
import pickle
import sys
import numpy
from rdkit import DataStructs, RDConfig
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import CompositeRun
from rdkit.ML.Data import DataUtils, SplitData, Stats
def ScreenModel(mdl, descs, data, picking=[1], indices=[], errorEstimate=0):
    """ collects the results of screening an individual composite model that match
      a particular value

     **Arguments**

       - mdl: the composite model

       - descs: a list of descriptor names corresponding to the data set

       - data: the data set, a list of points to be screened.

       - picking: (Optional) a list of values that are to be collected.
         For examples, if you want an enrichment plot for picking the values
         1 and 2, you'd having picking=[1,2].

      **Returns**

        a list of 4-tuples containing:

           - the id of the point

           - the true result (from the data set)

           - the predicted result

           - the confidence value for the prediction

    """
    mdl.SetInputOrder(descs)
    for j in range(len(mdl)):
        tmp = mdl.GetModel(j)
        if hasattr(tmp, '_trainIndices') and (not isinstance(tmp._trainIndices, dict)):
            tis = {}
            if hasattr(tmp, '_trainIndices'):
                for v in tmp._trainIndices:
                    tis[v] = 1
            tmp._trainIndices = tis
    res = []
    if mdl.GetQuantBounds():
        needsQuant = 1
    else:
        needsQuant = 0
    if not indices:
        indices = list(range(len(data)))
    nTrueActives = 0
    for i in indices:
        if errorEstimate:
            use = []
            for j in range(len(mdl)):
                tmp = mdl.GetModel(j)
                if not tmp._trainIndices.get(i, 0):
                    use.append(j)
        else:
            use = None
        pt = data[i]
        pred, conf = mdl.ClassifyExample(pt, onlyModels=use)
        if needsQuant:
            pt = mdl.QuantizeActivity(pt[:])
        trueRes = pt[-1]
        if trueRes in picking:
            nTrueActives += 1
        if pred in picking:
            res.append((pt[0], trueRes, pred, conf))
    return (nTrueActives, res)