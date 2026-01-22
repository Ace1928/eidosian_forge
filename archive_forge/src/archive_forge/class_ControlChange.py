from __future__ import print_function
from .utilities import key_number_to_key_name
class ControlChange(object):
    """A control change event.

    Parameters
    ----------
    number : int
        The control change number, in ``[0, 127]``.
    value : int
        The value of the control change, in ``[0, 127]``.
    time : float
        Time where the control change occurs.

    """

    def __init__(self, number, value, time):
        self.number = number
        self.value = value
        self.time = time

    def __repr__(self):
        return 'ControlChange(number={:d}, value={:d}, time={:f})'.format(self.number, self.value, self.time)