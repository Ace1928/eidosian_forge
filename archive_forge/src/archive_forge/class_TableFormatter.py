from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import io
import itertools
import json
import sys
import wcwidth
class TableFormatter(object):
    """Interface for table formatters."""
    _empty_output_meaningful = False

    def __init__(self, **kwds):
        """Initializes the base class.

    Keyword arguments:
      skip_header_when_empty: If true, does not print the table's header
        if there are zero rows. This argument has no effect on
        PrettyJsonFormatter. Ignored by the Print method, but respected if
        calling str or unicode on the formatter itself. Print will emit nothing
        if there are zero rows, unless the format being emitted requires text
        to be valid (eg json).
    """
        if self.__class__ == TableFormatter:
            raise NotImplementedError('Cannot instantiate abstract class TableFormatter')
        self.skip_header_when_empty = kwds.get('skip_header_when_empty', False)

    def __nonzero__(self):
        return bool(len(self))

    def __len__(self):
        raise NotImplementedError('__len__ must be implemented by subclass')

    def __str__(self):
        return self._EncodedStr(sys.getdefaultencoding())

    def __unicode__(self):
        raise NotImplementedError('__unicode__ must be implemented by subclass')

    def _EncodedStr(self, encoding):
        return self.__unicode__().encode(encoding, 'backslashreplace').decode(encoding)

    def Print(self, output=None):
        if self or self._empty_output_meaningful:
            file = output if output else sys.stdout
            encoding = sys.stdout.encoding or 'utf8'
            print(self._EncodedStr(encoding), file=file)

    def AddRow(self, row):
        """Add a new row (an iterable) to this formatter."""
        raise NotImplementedError('AddRow must be implemented by subclass')

    def AddRows(self, rows):
        """Add all rows to this table."""
        for row in rows:
            self.AddRow(row)

    def AddField(self, field):
        """Add a field as a new column to this formatter."""
        align = 'l' if field.get('type', []) == 'STRING' else 'r'
        self.AddColumn(field['name'], align=align)

    def AddFields(self, fields):
        """Convenience method to add a list of fields."""
        for field in fields:
            self.AddField(field)

    def AddDict(self, d):
        """Add a dict as a row by using column names as keys."""
        self.AddRow([d.get(name, '') for name in self.column_names])

    @property
    def column_names(self):
        """Return the ordered list of column names in self."""
        raise NotImplementedError('column_names must be implemented by subclass')

    def AddColumn(self, column_name, align='r', **kwds):
        """Add a new column to this formatter."""
        raise NotImplementedError('AddColumn must be implemented by subclass')

    def AddColumns(self, column_names, kwdss=None):
        """Add a series of columns to this formatter."""
        kwdss = kwdss or [{}] * len(column_names)
        for column_name, kwds in zip(column_names, kwdss):
            self.AddColumn(column_name, **kwds)