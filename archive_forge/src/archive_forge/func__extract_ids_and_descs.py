import re
import warnings
from itertools import chain
from xml.etree import ElementTree
from xml.sax.saxutils import XMLGenerator, escape
from Bio import BiopythonParserWarning
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _extract_ids_and_descs(raw_id, raw_desc):
    """Extract IDs, descriptions, and raw ID from raw values (PRIVATE).

    Given values of the ``Hit_id`` and ``Hit_def`` elements, this function
    returns a tuple of three elements: all IDs, all descriptions, and the
    BLAST-generated ID. The BLAST-generated ID is set to ``None`` if no
    BLAST-generated IDs are present.

    """
    ids = []
    descs = []
    blast_gen_id = raw_id
    if raw_id.startswith('gnl|BL_ORD_ID|'):
        id_desc_line = raw_desc
    else:
        id_desc_line = raw_id + ' ' + raw_desc
    id_desc_pairs = [re.split(_RE_ID_DESC_PATTERN, x, maxsplit=1) for x in re.split(_RE_ID_DESC_PAIRS_PATTERN, id_desc_line)]
    for pair in id_desc_pairs:
        if len(pair) != 2:
            pair.append('')
        ids.append(pair[0])
        descs.append(pair[1])
    return (ids, descs, blast_gen_id)