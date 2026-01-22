import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def _fill_table(self, save_config=True):
    """
        Re-draw the table from scratch, by clearing out the table's
        multi-column listbox; and then filling it in with values from
        ``self._rows``.  Note that any cell-, row-, or column-specific
        color configuration that has been done will be lost.  The
        selection will also be lost -- i.e., no row will be selected
        after this call completes.
        """
    self._mlb.delete(0, 'end')
    for i, row in enumerate(self._rows):
        if self._reprfunc is not None:
            row = [self._reprfunc(i, j, v) for j, v in enumerate(row)]
        self._mlb.insert('end', row)