from statsmodels.compat.python import lmap, lrange
from itertools import cycle, zip_longest
import csv
def as_latex_tabular(self, center=True, **fmt_dict):
    """Return string, the table as a LaTeX tabular environment.
        Note: will require the booktabs package."""
    fmt = self._get_fmt('latex', **fmt_dict)
    formatted_rows = []
    if center:
        formatted_rows.append('\\begin{center}')
    table_dec_above = fmt['table_dec_above'] or ''
    table_dec_below = fmt['table_dec_below'] or ''
    prev_aligns = None
    last = None
    for row in self + [last]:
        if row == last:
            aligns = None
        else:
            aligns = row.get_aligns('latex', **fmt)
        if aligns != prev_aligns:
            if prev_aligns:
                formatted_rows.append(table_dec_below)
                formatted_rows.append('\\end{tabular}')
            if aligns:
                formatted_rows.append('\\begin{tabular}{%s}' % aligns)
                if not prev_aligns:
                    formatted_rows.append(table_dec_above)
        if row != last:
            formatted_rows.append(row.as_string(output_format='latex', **fmt))
        prev_aligns = aligns
    if self.title:
        title = '%%\\caption{%s}' % self.title
        formatted_rows.append(title)
    if center:
        formatted_rows.append('\\end{center}')
    return '\n'.join(formatted_rows).replace('$$', ' ')