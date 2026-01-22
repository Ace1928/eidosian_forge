class NotPowerSeriesError(BaseHolonomicError):

    def __init__(self, holonomic, x0):
        self.holonomic = holonomic
        self.x0 = x0

    def __str__(self):
        s = 'A Power Series does not exists for '
        s += str(self.holonomic)
        s += ' about %s.' % self.x0
        return s