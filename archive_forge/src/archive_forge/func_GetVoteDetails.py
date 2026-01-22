import pickle
import numpy
from rdkit.ML.DecTree import CrossValidate, PruneTree
def GetVoteDetails(self):
    """ Returns the details of the last vote the forest conducted

      this will be an empty list if no voting has yet been done

    """
    return self.treeVotes