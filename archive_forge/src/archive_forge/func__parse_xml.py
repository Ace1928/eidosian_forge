import xml.sax
from boto import handler
from boto.emr import emrobject
from boto.resultset import ResultSet
from tests.compat import unittest
def _parse_xml(self, body, markers):
    rs = ResultSet(markers)
    h = handler.XmlHandler(rs, None)
    xml.sax.parseString(body, h)
    return rs