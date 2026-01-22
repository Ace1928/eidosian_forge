import torch
class _GreaterThanEq(Constraint):
    """
    Constrain to a real half line `[lower_bound, inf)`.
    """

    def __init__(self, lower_bound):
        self.lower_bound = lower_bound
        super().__init__()

    def check(self, value):
        return self.lower_bound <= value

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += f'(lower_bound={self.lower_bound})'
        return fmt_string