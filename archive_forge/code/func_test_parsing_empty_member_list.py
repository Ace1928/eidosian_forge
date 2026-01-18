from boto.mws.connection import MWSConnection
from boto.mws.response import (ResponseFactory, ResponseElement, Element,
from tests.unit import AWSMockServiceTestCase
from boto.compat import filter, map
from tests.compat import unittest
def test_parsing_empty_member_list(self):

    class Test5Result(ResponseElement):
        Item = MemberList(Nest=MemberList())
    text = b'<Test5Response><Test5Result>\n                  <Item/>\n                  </Test5Result></Test5Response>'
    obj = self.check_issue(Test5Result, text)
    self.assertSequenceEqual(obj._result.Item, [])