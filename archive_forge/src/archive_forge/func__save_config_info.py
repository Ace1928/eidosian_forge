import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def _save_config_info(self, row_indices=None, index_by_id=False):
    """
        Return a 'cookie' containing information about which row is
        selected, and what color configurations have been applied.
        this information can the be re-applied to the table (after
        making modifications) using ``_restore_config_info()``.  Color
        configuration information will be saved for any rows in
        ``row_indices``, or in the entire table, if
        ``row_indices=None``.  If ``index_by_id=True``, the the cookie
        will associate rows with their configuration information based
        on the rows' python id.  This is useful when performing
        operations that re-arrange the rows (e.g. ``sort``).  If
        ``index_by_id=False``, then it is assumed that all rows will be
        in the same order when ``_restore_config_info()`` is called.
        """
    if row_indices is None:
        row_indices = list(range(len(self._rows)))
    selection = self.selected_row()
    if index_by_id and selection is not None:
        selection = id(self._rows[selection])
    if index_by_id:
        config = {id(self._rows[r]): [self._get_itemconfig(r, c) for c in range(self._num_columns)] for r in row_indices}
    else:
        config = {r: [self._get_itemconfig(r, c) for c in range(self._num_columns)] for r in row_indices}
    return (selection, config)