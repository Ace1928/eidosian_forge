from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _get_simple_html_string(self, options):
    from html import escape
    lines = []
    if options['xhtml']:
        linebreak = '<br/>'
    else:
        linebreak = '<br>'
    open_tag = ['<table']
    if options['attributes']:
        for attr_name in options['attributes']:
            open_tag.append(f' {escape(attr_name)}="{escape(options['attributes'][attr_name])}"')
    open_tag.append('>')
    lines.append(''.join(open_tag))
    title = options['title'] or self._title
    if title:
        lines.append(f'    <caption>{escape(title)}</caption>')
    if options['header']:
        lines.append('    <thead>')
        lines.append('        <tr>')
        for field in self._field_names:
            if options['fields'] and field not in options['fields']:
                continue
            lines.append('            <th>%s</th>' % escape(field).replace('\n', linebreak))
        lines.append('        </tr>')
        lines.append('    </thead>')
    lines.append('    <tbody>')
    rows = self._get_rows(options)
    formatted_rows = self._format_rows(rows)
    for row in formatted_rows:
        lines.append('        <tr>')
        for field, datum in zip(self._field_names, row):
            if options['fields'] and field not in options['fields']:
                continue
            lines.append('            <td>%s</td>' % escape(datum).replace('\n', linebreak))
        lines.append('        </tr>')
    lines.append('    </tbody>')
    lines.append('</table>')
    return '\n'.join(lines)