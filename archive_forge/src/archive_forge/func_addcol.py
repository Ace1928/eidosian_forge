def addcol(self, col, rowitems):
    """adds a column"""
    if col in self.cols:
        for row, item in rowitems.items():
            self.add(row, col, item, colcheck=False)
    else:
        raise RuntimeError('col is not in the matrix columns')