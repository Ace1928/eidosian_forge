import re
from .enumerated import ENUM
from .enumerated import SET
from .types import DATETIME
from .types import TIME
from .types import TIMESTAMP
from ... import log
from ... import types as sqltypes
from ... import util
def _describe_to_create(self, table_name, columns):
    """Re-format DESCRIBE output as a SHOW CREATE TABLE string.

        DESCRIBE is a much simpler reflection and is sufficient for
        reflecting views for runtime use.  This method formats DDL
        for columns only- keys are omitted.

        :param columns: A sequence of DESCRIBE or SHOW COLUMNS 6-tuples.
          SHOW FULL COLUMNS FROM rows must be rearranged for use with
          this function.
        """
    buffer = []
    for row in columns:
        name, col_type, nullable, default, extra = (row[i] for i in (0, 1, 2, 4, 5))
        line = [' ']
        line.append(self.preparer.quote_identifier(name))
        line.append(col_type)
        if not nullable:
            line.append('NOT NULL')
        if default:
            if 'auto_increment' in default:
                pass
            elif col_type.startswith('timestamp') and default.startswith('C'):
                line.append('DEFAULT')
                line.append(default)
            elif default == 'NULL':
                line.append('DEFAULT')
                line.append(default)
            else:
                line.append('DEFAULT')
                line.append("'%s'" % default.replace("'", "''"))
        if extra:
            line.append(extra)
        buffer.append(' '.join(line))
    return ''.join(['CREATE TABLE %s (\n' % self.preparer.quote_identifier(table_name), ',\n'.join(buffer), '\n) '])