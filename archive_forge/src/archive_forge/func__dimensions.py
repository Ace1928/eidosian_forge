from functools import reduce
import copy
import math
import random
import sys
import warnings
from Bio import File
from Bio.Data import IUPACData
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning, BiopythonWarning
from Bio.Nexus.StandardData import StandardData
from Bio.Nexus.Trees import Tree
def _dimensions(self, options):
    if 'ntax' in options:
        self.ntax = eval(options['ntax'])
    if 'nchar' in options:
        self.nchar = eval(options['nchar'])