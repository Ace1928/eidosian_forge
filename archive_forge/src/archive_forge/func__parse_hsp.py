import re
from xml.etree import ElementTree
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _parse_hsp(self, root_hsp_elem, query_id, hit_id, query_seq=None):
    """Parse hsp (PRIVATE)."""
    if root_hsp_elem is None:
        root_hsp_elem = []
    for hsp_elem in root_hsp_elem:
        frag = HSPFragment(hit_id, query_id)
        setattr(frag, 'molecule_type', 'protein')
        if query_seq is not None:
            setattr(frag, 'query', query_seq)
        for key, (attr, caster) in _ELEM_FRAG.items():
            value = hsp_elem.attrib.get(key)
            if value is not None:
                if attr.endswith('start'):
                    value = caster(value) - 1
                if attr == 'query_start':
                    start = int(value)
                if attr == 'query_end':
                    end = int(value)
                setattr(frag, attr, caster(value))
        setattr(frag, 'aln_span', end - start)
        hsp = HSP([frag])
        setattr(hsp, 'query_id', query_id)
        setattr(hsp, 'hit_id', hit_id)
        for key, (attr, caster) in _ELEM_HSP.items():
            value = hsp_elem.attrib.get(key)
            if value is not None:
                setattr(hsp, attr, caster(value))
        yield hsp