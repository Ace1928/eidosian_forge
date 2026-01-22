from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _get_formatted_html_string(self, options):
    from html import escape
    lines = []
    lpad, rpad = self._get_padding_widths(options)
    if options['xhtml']:
        linebreak = '<br/>'
    else:
        linebreak = '<br>'
    open_tag = ['<table']
    if options['border']:
        if options['hrules'] == ALL and options['vrules'] == ALL:
            open_tag.append(' frame="box" rules="all"')
        elif options['hrules'] == FRAME and options['vrules'] == FRAME:
            open_tag.append(' frame="box"')
        elif options['hrules'] == FRAME and options['vrules'] == ALL:
            open_tag.append(' frame="box" rules="cols"')
        elif options['hrules'] == FRAME:
            open_tag.append(' frame="hsides"')
        elif options['hrules'] == ALL:
            open_tag.append(' frame="hsides" rules="rows"')
        elif options['vrules'] == FRAME:
            open_tag.append(' frame="vsides"')
        elif options['vrules'] == ALL:
            open_tag.append(' frame="vsides" rules="cols"')
    if not options['border'] and options['preserve_internal_border']:
        open_tag.append(' rules="cols"')
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
            lines.append('            <th style="padding-left: %dem; padding-right: %dem; text-align: center">%s</th>' % (lpad, rpad, escape(field).replace('\n', linebreak)))
        lines.append('        </tr>')
        lines.append('    </thead>')
    lines.append('    <tbody>')
    rows = self._get_rows(options)
    formatted_rows = self._format_rows(rows)
    aligns = []
    valigns = []
    for field in self._field_names:
        aligns.append({'l': 'left', 'r': 'right', 'c': 'center'}[self._align[field]])
        valigns.append({'t': 'top', 'm': 'middle', 'b': 'bottom'}[self._valign[field]])
    for row in formatted_rows:
        lines.append('        <tr>')
        for field, datum, align, valign in zip(self._field_names, row, aligns, valigns):
            if options['fields'] and field not in options['fields']:
                continue
            lines.append('            <td style="padding-left: %dem; padding-right: %dem; text-align: %s; vertical-align: %s">%s</td>' % (lpad, rpad, align, valign, escape(datum).replace('\n', linebreak)))
        lines.append('        </tr>')
    lines.append('    </tbody>')
    lines.append('</table>')
    return '\n'.join(lines)