import datetime
import re
import sys
import threading
import time
import traceback
import unittest
from xml.sax import saxutils
from absl.testing import _pretty_print_reporter
def _escape_xml_attr(content):
    """Escapes xml attributes."""
    return saxutils.escape(content, _escape_xml_attr_conversions)