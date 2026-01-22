import re
from math import log
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _list_from_csv(csv_string, caster=None):
    """Transform the given comma-separated string into a list (PRIVATE).

    :param csv_string: comma-separated input string
    :type csv_string: string
    :param caster: function used to cast each item in the input string
                   to its intended type
    :type caster: callable, accepts string, returns object

    """
    if caster is None:
        return [x for x in csv_string.split(',') if x]
    else:
        return [caster(x) for x in csv_string.split(',') if x]