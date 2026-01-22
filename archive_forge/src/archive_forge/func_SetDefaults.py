from warnings import warn
import pickle
import sys
import time
import numpy
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import BuildComposite, CompositeRun, ScreenComposite
from rdkit.ML.Composite import AdjustComposite
from rdkit.ML.Data import DataUtils, SplitData
def SetDefaults(runDetails=None):
    """  initializes a details object with default values

      **Arguments**

        - details:  (optional) a _CompositeRun.CompositeRun_ object.
          If this is not provided, the global _runDetails will be used.

      **Returns**

        the initialized _CompositeRun_ object.


  """
    if runDetails is None:
        runDetails = _runDetails
    return CompositeRun.SetDefaults(runDetails)