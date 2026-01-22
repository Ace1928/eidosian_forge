import os
import pickle
import sys
import numpy
from rdkit import RDConfig
from rdkit.Chem import FragmentCatalog
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import InfoTheory

     gains should be a sequence of sequences.  The idCol entry of each
     sub-sequence should be a catalog ID.  _ProcessGainsData()_ provides
     suitable input.

    