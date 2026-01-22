from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import datetime
import functools
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
from absl import app
from absl import flags
import yaml
import table_formatter
import bq_utils
from clients import utils as bq_client_utils
from utils import bq_error
from utils import bq_id_utils
from pyglib import stringutil
class TablePrinter(object):
    """Base class for printing a table, with a default implementation."""

    def __init__(self, **kwds):
        super(TablePrinter, self).__init__()
        for key, value in kwds.items():
            setattr(self, key, value)

    @staticmethod
    def _ValidateFields(fields, formatter):
        if isinstance(formatter, table_formatter.CsvFormatter):
            for field in fields:
                if field['type'].upper() == 'RECORD':
                    raise app.UsageError('Error printing table: Cannot print record field "%s" in CSV format.' % field['name'])
                if field.get('mode', 'NULLABLE').upper() == 'REPEATED':
                    raise app.UsageError('Error printing table: Cannot print repeated field "%s" in CSV format.' % field['name'])

    @staticmethod
    def _NormalizeRecord(field, value):
        """Returns bq-specific formatting of a RECORD type."""
        result = collections.OrderedDict()
        for subfield, subvalue in zip(field.get('fields', []), value):
            result[subfield.get('name', '')] = TablePrinter.NormalizeField(subfield, subvalue)
        return result

    @staticmethod
    def _NormalizeTimestamp(unused_field, value):
        """Returns bq-specific formatting of a TIMESTAMP type."""
        try:
            date = datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc) + datetime.timedelta(seconds=float(value))
            date = date.replace(tzinfo=None)
            date = date.replace(microsecond=0)
            return date.isoformat(' ')
        except ValueError:
            return '<date out of range for display>'

    @staticmethod
    def _NormalizeRange(field, value):
        """Returns bq-specific formatting of a RANGE type."""
        parsed = ParseRangeString(value)
        if parsed is None:
            return '<invalid range>'
        start, end = parsed
        if field.get('rangeElementType').get('type').upper() != 'TIMESTAMP':
            start = start.upper() if IsRangeBoundaryUnbounded(start) else start
            end = end.upper() if IsRangeBoundaryUnbounded(end) else end
            return '[%s, %s)' % (start, end)
        normalized_start = start.upper() if IsRangeBoundaryUnbounded(start) else TablePrinter._NormalizeTimestamp(field, start)
        normalized_end = end.upper() if IsRangeBoundaryUnbounded(end) else TablePrinter._NormalizeTimestamp(field, end)
        return '[%s, %s)' % (normalized_start, normalized_end)
    _FIELD_NORMALIZERS = {'RECORD': _NormalizeRecord.__func__, 'TIMESTAMP': _NormalizeTimestamp.__func__, 'RANGE': _NormalizeRange.__func__}

    @staticmethod
    def NormalizeField(field, value):
        """Returns bq-specific formatting of a field."""
        if value is None:
            return None
        normalizer = TablePrinter._FIELD_NORMALIZERS.get(field.get('type', '').upper(), lambda _, x: x)
        if field.get('mode', '').upper() == 'REPEATED':
            return [normalizer(field, value) for value in value]
        return normalizer(field, value)

    @staticmethod
    def MaybeConvertToJson(value):
        """Converts dicts and lists to JSON; returns everything else as-is."""
        if isinstance(value, dict) or isinstance(value, list):
            return json.dumps(value, separators=(',', ':'), ensure_ascii=False)
        return value

    @staticmethod
    def FormatRow(fields, row, formatter):
        """Convert fields in a single row to bq-specific formatting."""
        values = [TablePrinter.NormalizeField(field, value) for field, value in zip(fields, row)]
        if not isinstance(formatter, table_formatter.JsonFormatter):
            values = map(TablePrinter.MaybeConvertToJson, values)
        if isinstance(formatter, table_formatter.CsvFormatter):
            values = ['' if value is None else value for value in values]
        elif not isinstance(formatter, table_formatter.JsonFormatter):
            values = ['NULL' if value is None else value for value in values]
        return values

    def PrintTable(self, fields, rows):
        formatter = GetFormatterFromFlags(secondary_format='pretty')
        self._ValidateFields(fields, formatter)
        formatter.AddFields(fields)
        formatter.AddRows((TablePrinter.FormatRow(fields, row, formatter) for row in rows))
        formatter.Print()