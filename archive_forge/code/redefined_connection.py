
from lognormal_around import lognormal_around

class RedefinedConnection:
    """
    Redefined Connection class with specified parameters.
    """
    def __init__(self):
        self.strength = lognormal_around(2.75, 0.5, 5)
        self.delay = int(lognormal_around(3, 1, 5))
