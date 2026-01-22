from typing import NamedTuple
from typing import Sequence, Tuple
class ShotCopies(NamedTuple):
    """A namedtuple that represents a shot quantity being repeated some number of times.
    For example, ``ShotCopies(10 shots x 2)`` indicates two executions with 10 shots each for 20 shots total.
    """
    shots: int
    copies: int

    def __str__(self):
        """The string representation of the class"""
        return f'{self.shots} shots{(' x ' + str(self.copies) if self.copies > 1 else '')}'

    def __repr__(self):
        """The representation of the class"""
        return f'ShotCopies({self.shots} shots x {self.copies})'