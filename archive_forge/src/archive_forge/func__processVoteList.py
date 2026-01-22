from warnings import warn
import os
import pickle
import sys
import numpy
from rdkit import DataStructs
from rdkit.Dbase import DbModule
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import CompositeRun
from rdkit.ML.Data import DataUtils, SplitData
def _processVoteList(votes, data):
    """ *Internal Use Only*

  converts a list of 4 tuples: (answer,prediction,confidence,idx) into
  an alternate list: (answer,prediction,confidence,data point)

   **Arguments**

     - votes: a list of 4 tuples: (answer, prediction, confidence,
       index)

     - data: a _DataUtils.MLData.MLDataSet_


   **Note**: alterations are done in place in the _votes_ list

  """
    for i in range(len(votes)):
        ans, pred, conf, idx = votes[i]
        votes[i] = (ans, pred, conf, data[idx])