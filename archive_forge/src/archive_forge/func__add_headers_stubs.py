from statsmodels.compat.python import lmap, lrange
from itertools import cycle, zip_longest
import csv
def _add_headers_stubs(self, headers, stubs):
    """Return None.  Adds headers and stubs to table,
        if these were provided at initialization.
        Parameters
        ----------
        headers : list[str]
            K strings, where K is number of columns
        stubs : list[str]
            R strings, where R is number of non-header rows

        :note: a header row does not receive a stub!
        """
    if headers:
        self.insert_header_row(0, headers, dec_below='header_dec_below')
    if stubs:
        self.insert_stubs(0, stubs)