import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _parse_result_row(self):
    """Return a dictionary of parsed row values (PRIVATE)."""
    fields = self.fields
    columns = self.line.strip().split('\t')
    if len(fields) != len(columns):
        raise ValueError('Expected %i columns, found: %i' % (len(fields), len(columns)))
    qresult, hit, hsp, frag = ({}, {}, {}, {})
    for idx, value in enumerate(columns):
        sname = fields[idx]
        in_mapping = False
        for parsed_dict, mapping in ((qresult, _COLUMN_QRESULT), (hit, _COLUMN_HIT), (hsp, _COLUMN_HSP), (frag, _COLUMN_FRAG)):
            if sname in mapping:
                attr_name, caster = mapping[sname]
                if caster is not str:
                    value = caster(value)
                parsed_dict[attr_name] = value
                in_mapping = True
        if not in_mapping:
            assert sname not in _SUPPORTED_FIELDS
    return {'qresult': qresult, 'hit': hit, 'hsp': hsp, 'frag': frag}