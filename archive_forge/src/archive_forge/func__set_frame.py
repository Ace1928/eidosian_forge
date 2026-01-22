import re
from functools import reduce
from abc import ABC, abstractmethod
from typing import Optional, Type
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from Bio.SeqUtils import seq1
def _set_frame(frag):
    """Set the HSPFragment frames (PRIVATE)."""
    frag.hit_frame = (frag.hit_start % 3 + 1) * frag.hit_strand
    frag.query_frame = (frag.query_start % 3 + 1) * frag.query_strand