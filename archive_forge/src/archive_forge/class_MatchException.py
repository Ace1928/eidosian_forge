import os
import re
import six
from six.moves import urllib
from routes import request_config
class MatchException(RoutesException):
    """Tossed during URL matching exceptions"""