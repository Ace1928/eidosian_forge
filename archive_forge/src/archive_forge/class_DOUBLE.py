import datetime
from ... import exc
from ... import util
from ...sql import sqltypes
class DOUBLE(_FloatType, sqltypes.DOUBLE):
    """MySQL DOUBLE type."""
    __visit_name__ = 'DOUBLE'

    def __init__(self, precision=None, scale=None, asdecimal=True, **kw):
        """Construct a DOUBLE.

        .. note::

            The :class:`.DOUBLE` type by default converts from float
            to Decimal, using a truncation that defaults to 10 digits.
            Specify either ``scale=n`` or ``decimal_return_scale=n`` in order
            to change this scale, or ``asdecimal=False`` to return values
            directly as Python floating points.

        :param precision: Total digits in this number.  If scale and precision
          are both None, values are stored to limits allowed by the server.

        :param scale: The number of digits after the decimal point.

        :param unsigned: a boolean, optional.

        :param zerofill: Optional. If true, values will be stored as strings
          left-padded with zeros. Note that this does not effect the values
          returned by the underlying database API, which continue to be
          numeric.

        """
        super().__init__(precision=precision, scale=scale, asdecimal=asdecimal, **kw)