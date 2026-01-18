@property
def until_column(self):
    """The column where the error ends (starting with 0)."""
    return self._parso_error.end_pos[1]