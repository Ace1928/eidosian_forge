from boto.mws.connection import MWSConnection
from boto.mws.response import (ResponseFactory, ResponseElement, Element,
from tests.unit import AWSMockServiceTestCase
from boto.compat import filter, map
from tests.compat import unittest
class TestMWSResponse(AWSMockServiceTestCase):
    connection_class = MWSConnection
    mws = True

    def test_parsing_nested_elements(self):

        class Test9one(ResponseElement):
            Nest = Element()
            Zoom = Element()

        class Test9Result(ResponseElement):
            Item = Element(Test9one)
        text = b'<Test9Response><Test9Result>\n                  <Item>\n                        <Foo>Bar</Foo>\n                        <Nest>\n                            <Zip>Zap</Zip>\n                            <Zam>Zoo</Zam>\n                        </Nest>\n                        <Bif>Bam</Bif>\n                  </Item>\n                  </Test9Result></Test9Response>'
        obj = self.check_issue(Test9Result, text)
        Item = obj._result.Item
        useful = lambda x: not x[0].startswith('_')
        nest = dict(filter(useful, Item.Nest.__dict__.items()))
        self.assertEqual(nest, dict(Zip='Zap', Zam='Zoo'))
        useful = lambda x: not x[0].startswith('_') and (not x[0] == 'Nest')
        item = dict(filter(useful, Item.__dict__.items()))
        self.assertEqual(item, dict(Foo='Bar', Bif='Bam', Zoom=None))

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

    def test_parsing_member_list(self):

        class Test6Result(ResponseElement):
            Item = MemberList()
        text = b'<Test6Response><Test6Result>\n                  <Item>\n                        <member><Value>One</Value></member>\n                        <member><Value>Two</Value>\n                                <Error>Four</Error>\n                        </member>\n                        <member><Value>Six</Value></member>\n                  </Item>\n                  </Test6Result></Test6Response>'
        obj = self.check_issue(Test6Result, text)
        self.assertSequenceEqual([e.Value for e in obj._result.Item], ['One', 'Two', 'Six'])
        self.assertTrue(obj._result.Item[1].Error == 'Four')
        with self.assertRaises(AttributeError) as e:
            obj._result.Item[2].Error

    def test_parsing_empty_member_list(self):

        class Test5Result(ResponseElement):
            Item = MemberList(Nest=MemberList())
        text = b'<Test5Response><Test5Result>\n                  <Item/>\n                  </Test5Result></Test5Response>'
        obj = self.check_issue(Test5Result, text)
        self.assertSequenceEqual(obj._result.Item, [])

    def test_parsing_missing_member_list(self):

        class Test4Result(ResponseElement):
            Item = MemberList(NestedItem=MemberList())
        text = b'<Test4Response><Test4Result>\n                  </Test4Result></Test4Response>'
        obj = self.check_issue(Test4Result, text)
        self.assertSequenceEqual(obj._result.Item, [])

    def test_parsing_element_lists(self):

        class Test1Result(ResponseElement):
            Item = ElementList()
        text = b'<Test1Response><Test1Result>\n            <Item><Foo>Bar</Foo></Item>\n            <Item><Zip>Bif</Zip></Item>\n            <Item><Foo>Baz</Foo>\n                      <Zam>Zoo</Zam></Item>\n        </Test1Result></Test1Response>'
        obj = self.check_issue(Test1Result, text)
        self.assertTrue(len(obj._result.Item) == 3)
        elements = lambda x: getattr(x, 'Foo', getattr(x, 'Zip', '?'))
        elements = list(map(elements, obj._result.Item))
        self.assertSequenceEqual(elements, ['Bar', 'Bif', 'Baz'])

    def test_parsing_missing_lists(self):

        class Test2Result(ResponseElement):
            Item = ElementList()
        text = b'<Test2Response><Test2Result>\n        </Test2Result></Test2Response>'
        obj = self.check_issue(Test2Result, text)
        self.assertEqual(obj._result.Item, [])

    def test_parsing_simple_lists(self):

        class Test3Result(ResponseElement):
            Item = SimpleList()
        text = b'<Test3Response><Test3Result>\n            <Item>Bar</Item>\n            <Item>Bif</Item>\n            <Item>Baz</Item>\n        </Test3Result></Test3Response>'
        obj = self.check_issue(Test3Result, text)
        self.assertSequenceEqual(obj._result.Item, ['Bar', 'Bif', 'Baz'])

    def check_issue(self, klass, text):
        action = klass.__name__[:-len('Result')]
        factory = ResponseFactory(scopes=[{klass.__name__: klass}])
        parser = factory(action, connection=self.service_connection)
        return self.service_connection._parse_response(parser, 'text/xml', text)