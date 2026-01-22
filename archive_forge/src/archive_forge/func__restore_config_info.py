import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def _restore_config_info(self, cookie, index_by_id=False, see=False):
    """
        Restore selection & color configuration information that was
        saved using ``_save_config_info``.
        """
    selection, config = cookie
    if selection is None:
        self._mlb.selection_clear(0, 'end')
    if index_by_id:
        for r, row in enumerate(self._rows):
            if id(row) in config:
                for c in range(self._num_columns):
                    self._mlb.itemconfigure(r, c, config[id(row)][c])
            if id(row) == selection:
                self._mlb.select(r, see=see)
    else:
        if selection is not None:
            self._mlb.select(selection, see=see)
        for r in config:
            for c in range(self._num_columns):
                self._mlb.itemconfigure(r, c, config[r][c])