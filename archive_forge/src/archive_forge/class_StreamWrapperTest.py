from io import StringIO, TextIOWrapper
from unittest import TestCase, main
from ..ansitowin32 import AnsiToWin32, StreamWrapper
from ..win32 import ENABLE_VIRTUAL_TERMINAL_PROCESSING
from .utils import osname
class StreamWrapperTest(TestCase):

    def testIsAProxy(self):
        mockStream = Mock()
        wrapper = StreamWrapper(mockStream, None)
        self.assertTrue(wrapper.random_attr is mockStream.random_attr)

    def testDelegatesWrite(self):
        mockStream = Mock()
        mockConverter = Mock()
        wrapper = StreamWrapper(mockStream, mockConverter)
        wrapper.write('hello')
        self.assertTrue(mockConverter.write.call_args, (('hello',), {}))

    def testDelegatesContext(self):
        mockConverter = Mock()
        s = StringIO()
        with StreamWrapper(s, mockConverter) as fp:
            fp.write(u'hello')
        self.assertTrue(s.closed)

    def testProxyNoContextManager(self):
        mockStream = MagicMock()
        mockStream.__enter__.side_effect = AttributeError()
        mockConverter = Mock()
        with self.assertRaises(AttributeError) as excinfo:
            with StreamWrapper(mockStream, mockConverter) as wrapper:
                wrapper.write('hello')

    def test_closed_shouldnt_raise_on_closed_stream(self):
        stream = StringIO()
        stream.close()
        wrapper = StreamWrapper(stream, None)
        self.assertEqual(wrapper.closed, True)

    def test_closed_shouldnt_raise_on_detached_stream(self):
        stream = TextIOWrapper(StringIO())
        stream.detach()
        wrapper = StreamWrapper(stream, None)
        self.assertEqual(wrapper.closed, True)