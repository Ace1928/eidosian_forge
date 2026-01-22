from itertools import chain
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from .hmmer3_tab import Hmmer3TabParser, Hmmer3TabIndexer
class Hmmer3DomtabHmmqueryWriter(Hmmer3DomtabHmmhitWriter):
    """HMMER domain table writer using query coordinates.

    Writer for hmmer3-domtab output format which writes query coordinates
    as HMM profile coordinates.
    """
    hmm_as_hit = False