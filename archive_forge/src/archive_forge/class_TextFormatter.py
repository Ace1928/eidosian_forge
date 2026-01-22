import re
import html
from paste.util import PySourceColor
class TextFormatter(AbstractFormatter):

    def quote(self, s):
        return s

    def quote_long(self, s):
        return s

    def emphasize(self, s):
        return s

    def format_sup_object(self, obj):
        return 'In object: %s' % self.emphasize(self.quote(repr(obj)))

    def format_sup_url(self, url):
        return 'URL: %s' % self.quote(url)

    def format_sup_line_pos(self, line, column):
        if column:
            return self.emphasize('Line %i, Column %i' % (line, column))
        else:
            return self.emphasize('Line %i' % line)

    def format_sup_expression(self, expr):
        return self.emphasize('In expression: %s' % self.quote(expr))

    def format_sup_warning(self, warning):
        return 'Warning: %s' % self.quote(warning)

    def format_sup_info(self, info):
        return [self.quote_long(info)]

    def format_source_line(self, filename, frame):
        return 'File %r, line %s in %s' % (filename, frame.lineno or '?', frame.name or '?')

    def format_long_source(self, source, long_source):
        return self.format_source(source)

    def format_source(self, source_line):
        return '  ' + self.quote(source_line.strip())

    def format_exception_info(self, etype, evalue):
        return self.emphasize('%s: %s' % (self.quote(etype), self.quote(evalue)))

    def format_traceback_info(self, info):
        return info

    def format_combine(self, data_by_importance, lines, exc_info):
        lines[:0] = [value for n, value in data_by_importance['important']]
        lines.append(exc_info)
        for name in ('normal', 'supplemental', 'extra'):
            lines.extend([value for n, value in data_by_importance[name]])
        return self.format_combine_lines(lines)

    def format_combine_lines(self, lines):
        return '\n'.join(lines)

    def format_extra_data(self, importance, title, value):
        if isinstance(value, str):
            s = self.pretty_string_repr(value)
            if '\n' in s:
                return '%s:\n%s' % (title, s)
            else:
                return '%s: %s' % (title, s)
        elif isinstance(value, dict):
            lines = ['\n', title, '-' * len(title)]
            items = value.items()
            items = sorted(items)
            for n, v in items:
                try:
                    v = repr(v)
                except Exception as e:
                    v = 'Cannot display: %s' % e
                v = truncate(v)
                lines.append('  %s: %s' % (n, v))
            return '\n'.join(lines)
        elif isinstance(value, (list, tuple)) and self.long_item_list(value):
            parts = [truncate(repr(v)) for v in value]
            return '%s: [\n    %s]' % (title, ',\n    '.join(parts))
        else:
            return '%s: %s' % (title, truncate(repr(value)))