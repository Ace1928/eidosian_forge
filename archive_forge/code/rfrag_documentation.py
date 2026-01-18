import re
import sys
from rdkit import Chem
from rdkit.Chem import rdMMPA

optimization work.

Original:
~/RDKit_git/Contrib/mmpa > time head -100 ../../Data/Zinc/zim.smi | python rfrag.py > zim.frags.o

real	0m9.752s
user	0m9.704s
sys	0m0.043s


