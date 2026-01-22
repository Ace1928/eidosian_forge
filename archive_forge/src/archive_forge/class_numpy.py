from collections import defaultdict
import re
from .pyutil import memoize
from .periodic import symbols
class numpy:
    array = staticmethod(_numpy_not_installed_raise)
    log = staticmethod(_numpy_not_installed_raise)
    exp = staticmethod(_numpy_not_installed_raise)