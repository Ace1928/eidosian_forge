from warnings import warn
import pickle
import sys
import time
import numpy
from rdkit import DataStructs
from rdkit.Dbase import DbModule
from rdkit.ML import CompositeRun, ScreenComposite
from rdkit.ML.Composite import BayesComposite, Composite
from rdkit.ML.Data import DataUtils, SplitData
from rdkit.utils import listutils
 parses command line arguments and updates _runDetails_

      **Arguments**

        - runDetails:  a _CompositeRun.CompositeRun_ object.

  