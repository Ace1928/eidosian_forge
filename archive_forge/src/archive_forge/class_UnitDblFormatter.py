import matplotlib.ticker as ticker
class UnitDblFormatter(ticker.ScalarFormatter):
    """
    The formatter for UnitDbl data types.

    This allows for formatting with the unit string.
    """

    def __call__(self, x, pos=None):
        if len(self.locs) == 0:
            return ''
        else:
            return f'{x:.12}'

    def format_data_short(self, value):
        return f'{value:.12}'

    def format_data(self, value):
        return f'{value:.12}'