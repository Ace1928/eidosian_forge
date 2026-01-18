from boto.mws.connection import MWSConnection
from boto.mws.response import (ResponseFactory, ResponseElement, Element,
from tests.unit import AWSMockServiceTestCase
from boto.compat import filter, map
from tests.compat import unittest
def test_parsing_member_list_specification(self):

    class Test8extra(ResponseElement):
        Foo = SimpleList()

    class Test8Result(ResponseElement):
        Item = MemberList(SimpleList)
        Extra = MemberList(Test8extra)
    text = b'<Test8Response><Test8Result>\n                  <Item>\n                        <member>0</member>\n                        <member>1</member>\n                        <member>2</member>\n                        <member>3</member>\n                  </Item>\n                  <Extra>\n                        <member><Foo>4</Foo><Foo>5</Foo></member>\n                        <member></member>\n                        <member><Foo>6</Foo><Foo>7</Foo></member>\n                  </Extra>\n                  </Test8Result></Test8Response>'
    obj = self.check_issue(Test8Result, text)
    self.assertSequenceEqual(list(map(int, obj._result.Item)), list(range(4)))
    self.assertSequenceEqual(list(map(lambda x: list(map(int, x.Foo)), obj._result.Extra)), [[4, 5], [], [6, 7]])