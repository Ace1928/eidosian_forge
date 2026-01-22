from typing import Tuple as tTuple
from collections import defaultdict
from functools import cmp_to_key, reduce
from itertools import product
import operator
from .sympify import sympify
from .basic import Basic
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .cache import cacheit
from .logic import fuzzy_not, _fuzzy_group
from .expr import Expr
from .parameters import global_parameters
from .kind import KindDispatcher
from .traversal import bottom_up
from sympy.utilities.iterables import sift
from .numbers import Rational
from .power import Pow
from .add import Add, _unevaluated_Add
@staticmethod
def _matches_add_wildcard(dictionary, state):
    node_ind, target_ind = state
    if node_ind in dictionary:
        begin, end = dictionary[node_ind]
        dictionary[node_ind] = (begin, target_ind)
    else:
        dictionary[node_ind] = (target_ind, target_ind)