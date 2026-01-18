from datetime import timedelta
from decimal import Decimal, ROUND_FLOOR
def totimedelta(self, start=None, end=None):
    """
        Convert this duration into a timedelta object.

        This method requires a start datetime or end datetimem, but raises
        an exception if both are given.
        """
    if start is None and end is None:
        raise ValueError('start or end required')
    if start is not None and end is not None:
        raise ValueError('only start or end allowed')
    if start is not None:
        return start + self - start
    return end - (end - self)