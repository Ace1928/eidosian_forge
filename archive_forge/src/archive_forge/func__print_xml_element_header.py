import datetime
import re
import sys
import threading
import time
import traceback
import unittest
from xml.sax import saxutils
from absl.testing import _pretty_print_reporter
def _print_xml_element_header(element, attributes, stream, indentation=''):
    """Prints an XML header of an arbitrary element.

  Args:
    element: element name (testsuites, testsuite, testcase)
    attributes: 2-tuple list with (attributes, values) already escaped
    stream: output stream to write test report XML to
    indentation: indentation added to the element header
  """
    stream.write('%s<%s' % (indentation, element))
    for attribute in attributes:
        if len(attribute) == 2 and attribute[0] is not None and (attribute[1] is not None):
            stream.write(' %s="%s"' % (attribute[0], attribute[1]))
    stream.write('>\n')