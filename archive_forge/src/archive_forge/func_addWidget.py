from ..Qt import QtWidgets
def addWidget(self, item, row=None, col=None, rowspan=1, colspan=1):
    """
        Add a widget to the layout and place it in the next available cell (or in the cell specified).
        """
    if row == 'next':
        self.nextRow()
        row = self.currentRow
    elif row is None:
        row = self.currentRow
    if col is None:
        col = self.nextCol(colspan)
    if row not in self.rows:
        self.rows[row] = {}
    self.rows[row][col] = item
    self.items[item] = (row, col)
    self.layout.addWidget(item, row, col, rowspan, colspan)