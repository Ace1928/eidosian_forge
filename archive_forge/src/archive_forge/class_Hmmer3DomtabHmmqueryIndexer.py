from itertools import chain
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from .hmmer3_tab import Hmmer3TabParser, Hmmer3TabIndexer
class Hmmer3DomtabHmmqueryIndexer(Hmmer3TabIndexer):
    """HMMER domain table indexer using query coordinates.

    Indexer class for HMMER domain table output that assumes HMM profile
    coordinates are query coordinates.
    """
    _parser = Hmmer3DomtabHmmqueryParser
    _query_id_idx = 3