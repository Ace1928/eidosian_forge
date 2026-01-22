import os
import re
from collections import namedtuple
from contextlib import closing
from rdkit import Chem, RDConfig
from rdkit.Chem.rdmolfiles import SDMolSupplier, SmilesMolSupplier


        >>> remover = SaltRemover(defnData="[Cl,Br]")
        >>> len(remover.salts)
        1
        >>> Chem.MolToSmarts(remover.salts[0])
        '[Cl,Br]'

        >>> mol = Chem.MolFromSmiles('CN(C)C.Cl')
        >>> res = remover(mol)
        >>> res is not None
        True
        >>> res.GetNumAtoms()
        4

        