from warnings import warn
import pickle
import sys
import numpy
from rdkit import DataStructs, RDConfig
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import CompositeRun
from rdkit.ML.Data import DataUtils, SplitData, Stats
def AccumulateCounts(predictions, thresh=0, sortIt=1):
    """  Accumulates the data for the enrichment plot for a single model

      **Arguments**

        - predictions: a list of 3-tuples (as returned by _ScreenModels_)

        - thresh: a threshold for the confidence level.  Anything below
          this threshold will not be considered

        - sortIt: toggles sorting on confidence levels


      **Returns**

        - a list of 3-tuples:

          - the id of the active picked here

          - num actives found so far

          - number of picks made so far

    """
    if sortIt:
        predictions.sort(lambda x, y: cmp(y[3], x[3]))
    res = []
    nCorrect = 0
    nPts = 0
    for i in range(len(predictions)):
        ID, real, pred, conf = predictions[i]
        if conf > thresh:
            if pred == real:
                nCorrect += 1
            nPts += 1
            res.append((ID, nCorrect, nPts))
    return res