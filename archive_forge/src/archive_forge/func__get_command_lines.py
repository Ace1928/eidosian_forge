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
def _get_command_lines(file_contents):
    lines = _kill_comments_and_break_lines(file_contents)
    commandlines = _adjust_lines(lines)
    return commandlines