from boto.mws.connection import MWSConnection
from boto.mws.response import (ResponseFactory, ResponseElement, Element,
from tests.unit import AWSMockServiceTestCase
from boto.compat import filter, map
from tests.compat import unittest
def test_parsing_missing_lists(self):

    class Test2Result(ResponseElement):
        Item = ElementList()
    text = b'<Test2Response><Test2Result>\n        </Test2Result></Test2Response>'
    obj = self.check_issue(Test2Result, text)
    self.assertEqual(obj._result.Item, [])