from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import tzinfo
from zope.interface import Attribute
from zope.interface import Interface
from zope.interface import classImplements
class ITimeDeltaClass(Interface):
    """This is the timedelta class interface.

    This is symbolic; this module does **not** make
    `datetime.timedelta` provide this interface.
    """
    min = Attribute('The most negative timedelta object')
    max = Attribute('The most positive timedelta object')
    resolution = Attribute('The smallest difference between non-equal timedelta objects')