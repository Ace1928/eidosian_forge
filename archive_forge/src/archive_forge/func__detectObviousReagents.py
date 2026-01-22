import itertools
from collections import Counter, defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdqueries
from . import utils
def _detectObviousReagents(reactants, products):
    unchangedReacts = set()
    unchangedProds = set()
    for i, r in enumerate(reactants):
        for j, p in enumerate(products):
            if r == p:
                unchangedReacts.add(i)
                unchangedProds.add(j)
    return (unchangedReacts, unchangedProds)