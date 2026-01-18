import datetime
import xml.sax
import unittest
import boto.handler
import boto.resultset
import boto.cloudformation
def test_resource_time_with_millis(self):
    rs = boto.resultset.ResultSet([('member', boto.cloudformation.stack.StackResource)])
    h = boto.handler.XmlHandler(rs, None)
    xml.sax.parseString(DESCRIBE_STACK_RESOURCE_XML, h)
    timestamp_1 = rs[0].timestamp
    self.assertEqual(timestamp_1, datetime.datetime(2010, 7, 27, 22, 27, 28))
    timestamp_2 = rs[1].timestamp
    self.assertEqual(timestamp_2, datetime.datetime(2010, 7, 27, 22, 28, 28, 123456))