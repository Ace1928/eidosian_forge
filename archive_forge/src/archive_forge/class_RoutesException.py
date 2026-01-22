import os
import re
import six
from six.moves import urllib
from routes import request_config
class RoutesException(Exception):
    """Tossed during Route exceptions"""