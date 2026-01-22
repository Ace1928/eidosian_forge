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
def _adjust_lines(lines):
    """Adjust linebreaks to match ';', strip leading/trailing whitespace (PRIVATE).

    list_of_commandlines=_adjust_lines(input_text)
    Lines are adjusted so that no linebreaks occur within a commandline
    (except matrix command line)
    """
    formatted_lines = []
    for line in lines:
        line = line.replace('\r\n', '\n').replace('\r', '\n').strip()
        if line.lower().startswith('matrix'):
            formatted_lines.append(line)
        else:
            line = line.replace('\n', ' ')
            if line:
                formatted_lines.append(line)
    return formatted_lines