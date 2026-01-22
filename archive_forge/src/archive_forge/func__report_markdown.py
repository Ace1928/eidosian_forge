from __future__ import annotations
import sys
from typing import Any, IO, Iterable, TYPE_CHECKING
from coverage.exceptions import ConfigError, NoDataError
from coverage.misc import human_sorted_items
from coverage.plugin import FileReporter
from coverage.report_core import get_analysis_to_report
from coverage.results import Analysis, Numbers
from coverage.types import TMorf
def _report_markdown(self, header: list[str], lines_values: list[list[Any]], total_line: list[Any], end_lines: list[str]) -> None:
    """Internal method that prints report data in markdown format.

        `header` is a list with captions.
        `lines_values` is a sorted list of lists containing coverage information.
        `total_line` is a list with values of the total line.
        `end_lines` is a list of ending lines with information about skipped files.

        """
    max_name = max((len(line[0].replace('_', '\\_')) for line in lines_values), default=0)
    max_name = max(max_name, len('**TOTAL**')) + 1
    formats = dict(Name='| {:{name_len}}|', Stmts='{:>9} |', Miss='{:>9} |', Branch='{:>9} |', BrPart='{:>9} |', Cover='{:>{n}} |', Missing='{:>10} |')
    max_n = max(len(total_line[header.index('Cover')]) + 6, len(' Cover '))
    header_items = [formats[item].format(item, name_len=max_name, n=max_n) for item in header]
    header_str = ''.join(header_items)
    rule_str = '|' + ' '.join(['- |'.rjust(len(header_items[0]) - 1, '-')] + ['-: |'.rjust(len(item) - 1, '-') for item in header_items[1:]])
    self.write(header_str)
    self.write(rule_str)
    for values in lines_values:
        formats.update(dict(Cover='{:>{n}}% |'))
        line_items = [formats[item].format(str(value).replace('_', '\\_'), name_len=max_name, n=max_n - 1) for item, value in zip(header, values)]
        self.write_items(line_items)
    formats.update(dict(Name='|{:>{name_len}} |', Cover='{:>{n}} |'))
    total_line_items: list[str] = []
    for item, value in zip(header, total_line):
        if value == '':
            insert = value
        elif item == 'Cover':
            insert = f' **{value}%**'
        else:
            insert = f' **{value}**'
        total_line_items += formats[item].format(insert, name_len=max_name, n=max_n)
    self.write_items(total_line_items)
    for end_line in end_lines:
        self.write(end_line)