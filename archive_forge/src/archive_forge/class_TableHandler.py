from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
class TableHandler(HTMLParser):

    def __init__(self, **kwargs) -> None:
        HTMLParser.__init__(self)
        self.kwargs = kwargs
        self.tables: list[list] = []
        self.last_row: list[str] = []
        self.rows: list[Any] = []
        self.max_row_width = 0
        self.active = None
        self.last_content = ''
        self.is_last_row_header = False
        self.colspan = 0

    def handle_starttag(self, tag, attrs) -> None:
        self.active = tag
        if tag == 'th':
            self.is_last_row_header = True
        for key, value in attrs:
            if key == 'colspan':
                self.colspan = int(value)

    def handle_endtag(self, tag) -> None:
        if tag in ['th', 'td']:
            stripped_content = self.last_content.strip()
            self.last_row.append(stripped_content)
            if self.colspan:
                for i in range(1, self.colspan):
                    self.last_row.append('')
                self.colspan = 0
        if tag == 'tr':
            self.rows.append((self.last_row, self.is_last_row_header))
            self.max_row_width = max(self.max_row_width, len(self.last_row))
            self.last_row = []
            self.is_last_row_header = False
        if tag == 'table':
            table = self.generate_table(self.rows)
            self.tables.append(table)
            self.rows = []
        self.last_content = ' '
        self.active = None

    def handle_data(self, data) -> None:
        self.last_content += data

    def generate_table(self, rows):
        """
        Generates from a list of rows a PrettyTable object.
        """
        table = PrettyTable(**self.kwargs)
        for row in self.rows:
            if len(row[0]) < self.max_row_width:
                appends = self.max_row_width - len(row[0])
                for i in range(1, appends):
                    row[0].append('-')
            if row[1]:
                self.make_fields_unique(row[0])
                table.field_names = row[0]
            else:
                table.add_row(row[0])
        return table

    def make_fields_unique(self, fields) -> None:
        """
        iterates over the row and make each field unique
        """
        for i in range(0, len(fields)):
            for j in range(i + 1, len(fields)):
                if fields[i] == fields[j]:
                    fields[j] += "'"