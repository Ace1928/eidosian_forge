from boto.mws.connection import MWSConnection
from boto.mws.response import (ResponseFactory, ResponseElement, Element,
from tests.unit import AWSMockServiceTestCase
from boto.compat import filter, map
from tests.compat import unittest
def test_parsing_missing_member_list(self):

    class Test4Result(ResponseElement):
        Item = MemberList(NestedItem=MemberList())
    text = b'<Test4Response><Test4Result>\n                  </Test4Result></Test4Response>'
    obj = self.check_issue(Test4Result, text)
    self.assertSequenceEqual(obj._result.Item, [])