import csv
import difflib
from io import StringIO
from lxml import etree
from .jsonutil import JsonTable, get_column, get_where, get_selection
from .errors import is_xnat_error, catch_error
from .errors import ProgrammingError, NotSupportedError
from .errors import DataError, DatabaseError
def use_template(self, name, values):
    """
            Performs a search query using a previously saved template.

            Parameters
            ----------
            name: string
                Name of the template.
            values: dict
                Values to put in the template, get the valid keys using
                the get_template method.

            Examples
            --------
            >>> interface.manage.search.use_template(name,
                          {'subject_id':'ID',
                           'age':'32'
                           })

        """
    self._intf._get_entry_point()
    bundle = self.get_template(name, True) % values
    _query = query_from_xml(bundle)
    bundle = build_search_document(_query['row'], _query['columns'], _query['constraints'])
    content = self._intf._exec('%s/search?format=csv' % self._intf._entry, 'POST', bundle)
    results = csv.reader(StringIO(content), delimiter=',', quotechar='"')
    headers = results.next()
    return JsonTable([dict(zip(headers, res)) for res in results], headers)