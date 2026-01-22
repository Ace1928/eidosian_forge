from copy import deepcopy
from itertools import chain
from Bio.SearchIO._utils import optionalcascade
from ._base import _BaseSearchObject
from .hit import Hit
def _hit_key_func(hit):
    """Map hit to its identifier (PRIVATE).

    Default hit key function for QueryResult.__init__ use.
    """
    return hit.id