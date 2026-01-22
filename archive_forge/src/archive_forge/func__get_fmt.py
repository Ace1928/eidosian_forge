from statsmodels.compat.python import lmap, lrange
from itertools import cycle, zip_longest
import csv
def _get_fmt(self, output_format, **fmt_dict):
    """Return dict, the formatting options.
        """
    output_format = get_output_format(output_format)
    try:
        fmt = default_fmts[output_format].copy()
    except KeyError:
        raise ValueError('Unknown format: %s' % output_format)
    try:
        fmt.update(self.row.table.output_formats[output_format])
    except AttributeError:
        pass
    try:
        fmt.update(self.row._fmt)
    except AttributeError:
        pass
    fmt.update(self._fmt)
    fmt.update(fmt_dict)
    return fmt