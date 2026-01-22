import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def _gen_random_array(n):
    """Return an array of n random numbers summing to 1.0 (PRIVATE)."""
    randArray = [random.random() for _ in range(n)]
    total = sum(randArray)
    return [x / total for x in randArray]