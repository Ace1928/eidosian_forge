import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _parse_fields_line(self):
    """Return column short names line from 'Fields' comment line (PRIVATE)."""
    raw_field_str = self.line[len('# Fields: '):]
    long_fields = raw_field_str.split(', ')
    fields = [_LONG_SHORT_MAP[long_name] for long_name in long_fields]
    return self._prep_fields(fields)