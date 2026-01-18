from boto.mws.connection import MWSConnection
from boto.mws.response import (ResponseFactory, ResponseElement, Element,
from tests.unit import AWSMockServiceTestCase
from boto.compat import filter, map
from tests.compat import unittest
def test_parsing_nested_lists(self):

    class Test7Result(ResponseElement):
        Item = MemberList(Nest=MemberList(), List=ElementList(Simple=SimpleList()))
    text = b'<Test7Response><Test7Result>\n                  <Item>\n                        <member>\n                            <Value>One</Value>\n                            <Nest>\n                                <member><Data>2</Data></member>\n                                <member><Data>4</Data></member>\n                                <member><Data>6</Data></member>\n                            </Nest>\n                        </member>\n                        <member>\n                            <Value>Two</Value>\n                            <Nest>\n                                <member><Data>1</Data></member>\n                                <member><Data>3</Data></member>\n                                <member><Data>5</Data></member>\n                            </Nest>\n                            <List>\n                                <Simple>4</Simple>\n                                <Simple>5</Simple>\n                                <Simple>6</Simple>\n                            </List>\n                            <List>\n                                <Simple>7</Simple>\n                                <Simple>8</Simple>\n                                <Simple>9</Simple>\n                            </List>\n                        </member>\n                        <member>\n                            <Value>Six</Value>\n                            <List>\n                                <Complex>Foo</Complex>\n                                <Simple>1</Simple>\n                                <Simple>2</Simple>\n                                <Simple>3</Simple>\n                            </List>\n                            <List>\n                                <Complex>Bar</Complex>\n                            </List>\n                        </member>\n                  </Item>\n                  </Test7Result></Test7Response>'
    obj = self.check_issue(Test7Result, text)
    item = obj._result.Item
    self.assertEqual(len(item), 3)
    nests = [z.Nest for z in filter(lambda x: x.Nest, item)]
    self.assertSequenceEqual([[y.Data for y in nest] for nest in nests], [[u'2', u'4', u'6'], [u'1', u'3', u'5']])
    self.assertSequenceEqual([element.Simple for element in item[1].List], [[u'4', u'5', u'6'], [u'7', u'8', u'9']])
    self.assertSequenceEqual(item[-1].List[0].Simple, ['1', '2', '3'])
    self.assertEqual(item[-1].List[1].Simple, [])
    self.assertSequenceEqual([e.Value for e in obj._result.Item], ['One', 'Two', 'Six'])