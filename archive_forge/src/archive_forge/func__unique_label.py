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
def _unique_label(previous_labels, label):
    """Return a unique name if label is already in previous_labels (PRIVATE)."""
    while label in previous_labels:
        label_split = label.split('.')
        if label_split[-1].startswith('copy'):
            copy_num = 1
            if label_split[-1] != 'copy':
                copy_num = int(label_split[-1][4:]) + 1
            new_label = f'{'.'.join(label_split[:-1])}.copy{copy_num}'
            label = new_label
        else:
            label += '.copy'
    return label