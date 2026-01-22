import base64
import sys
import time
from datetime import datetime
from decimal import Decimal
import http.client
import urllib.parse
from xml.parsers import expat
import errno
from io import BytesIO
def _datetime_type(data):
    return datetime.strptime(data, '%Y%m%dT%H:%M:%S')