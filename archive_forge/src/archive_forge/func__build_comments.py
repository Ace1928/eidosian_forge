import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _build_comments(self, qres):
    """Return QueryResult tabular comment as a string (PRIVATE)."""
    comments = []
    inv_field_map = {v: k for k, v in _LONG_SHORT_MAP.items()}
    program = qres.program.upper()
    try:
        version = qres.version
    except AttributeError:
        program_line = '# %s' % program
    else:
        program_line = f'# {program} {version}'
    comments.append(program_line)
    if qres.description is None:
        comments.append('# Query: %s' % qres.id)
    else:
        comments.append(f'# Query: {qres.id} {qres.description}')
    try:
        comments.append('# RID: %s' % qres.rid)
    except AttributeError:
        pass
    comments.append('# Database: %s' % qres.target)
    if qres:
        comments.append('# Fields: %s' % ', '.join((inv_field_map[field] for field in self.fields)))
    comments.append('# %i hits found' % len(qres))
    return '\n'.join(comments) + '\n'