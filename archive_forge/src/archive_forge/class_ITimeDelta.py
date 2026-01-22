from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import tzinfo
from zope.interface import Attribute
from zope.interface import Interface
from zope.interface import classImplements
class ITimeDelta(ITimeDeltaClass):
    """Represent the difference between two datetime objects.

    Implemented by `datetime.timedelta`.

    Supported operators:

    - add, subtract timedelta
    - unary plus, minus, abs
    - compare to timedelta
    - multiply, divide by int/long

    In addition, `.datetime` supports subtraction of two `.datetime` objects
    returning a `.timedelta`, and addition or subtraction of a `.datetime`
    and a `.timedelta` giving a `.datetime`.

    Representation: (days, seconds, microseconds).
    """
    days = Attribute('Days between -999999999 and 999999999 inclusive')
    seconds = Attribute('Seconds between 0 and 86399 inclusive')
    microseconds = Attribute('Microseconds between 0 and 999999 inclusive')