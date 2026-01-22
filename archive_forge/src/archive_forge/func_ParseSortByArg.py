from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def ParseSortByArg(sort_by=None):
    """Parses and creates the sort by object from parsed arguments.

  Args:
    sort_by: list of strings, passed in from the --sort-by flag.

  Returns:
    A parsed sort by string ending in asc or desc, conforming to
    https://aip.dev/132#ordering
  """
    if not sort_by:
        return None
    fields = []
    for field in sort_by:
        if field.startswith('~'):
            field = field.lstrip('~') + ' desc'
        else:
            field += ' asc'
        fields.append(field)
    return ','.join(fields)