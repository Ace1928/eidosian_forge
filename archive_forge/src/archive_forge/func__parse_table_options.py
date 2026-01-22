import re
from .enumerated import ENUM
from .enumerated import SET
from .types import DATETIME
from .types import TIME
from .types import TIMESTAMP
from ... import log
from ... import types as sqltypes
from ... import util
def _parse_table_options(self, line, state):
    """Build a dictionary of all reflected table-level options.

        :param line: The final line of SHOW CREATE TABLE output.
        """
    options = {}
    if line and line != ')':
        rest_of_line = line
        for regex, cleanup in self._pr_options:
            m = regex.search(rest_of_line)
            if not m:
                continue
            directive, value = (m.group('directive'), m.group('val'))
            if cleanup:
                value = cleanup(value)
            options[directive.lower()] = value
            rest_of_line = regex.sub('', rest_of_line)
    for nope in ('auto_increment', 'data directory', 'index directory'):
        options.pop(nope, None)
    for opt, val in options.items():
        state.table_options['%s_%s' % (self.dialect.name, opt)] = val