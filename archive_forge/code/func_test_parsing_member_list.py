from boto.mws.connection import MWSConnection
from boto.mws.response import (ResponseFactory, ResponseElement, Element,
from tests.unit import AWSMockServiceTestCase
from boto.compat import filter, map
from tests.compat import unittest
def test_parsing_member_list(self):

    class Test6Result(ResponseElement):
        Item = MemberList()
    text = b'<Test6Response><Test6Result>\n                  <Item>\n                        <member><Value>One</Value></member>\n                        <member><Value>Two</Value>\n                                <Error>Four</Error>\n                        </member>\n                        <member><Value>Six</Value></member>\n                  </Item>\n                  </Test6Result></Test6Response>'
    obj = self.check_issue(Test6Result, text)
    self.assertSequenceEqual([e.Value for e in obj._result.Item], ['One', 'Two', 'Six'])
    self.assertTrue(obj._result.Item[1].Error == 'Four')
    with self.assertRaises(AttributeError) as e:
        obj._result.Item[2].Error