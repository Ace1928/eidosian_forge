import torch
class _Boolean(Constraint):
    """
    Constrain to the two values `{0, 1}`.
    """
    is_discrete = True

    def check(self, value):
        return (value == 0) | (value == 1)