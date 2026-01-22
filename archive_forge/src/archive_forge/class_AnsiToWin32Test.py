from io import StringIO, TextIOWrapper
from unittest import TestCase, main
from ..ansitowin32 import AnsiToWin32, StreamWrapper
from ..win32 import ENABLE_VIRTUAL_TERMINAL_PROCESSING
from .utils import osname
class AnsiToWin32Test(TestCase):

    def testInit(self):
        mockStdout = Mock()
        auto = Mock()
        stream = AnsiToWin32(mockStdout, autoreset=auto)
        self.assertEqual(stream.wrapped, mockStdout)
        self.assertEqual(stream.autoreset, auto)

    @patch('colorama.ansitowin32.winterm', None)
    @patch('colorama.ansitowin32.winapi_test', lambda *_: True)
    def testStripIsTrueOnWindows(self):
        with osname('nt'):
            mockStdout = Mock()
            stream = AnsiToWin32(mockStdout)
            self.assertTrue(stream.strip)

    def testStripIsFalseOffWindows(self):
        with osname('posix'):
            mockStdout = Mock(closed=False)
            stream = AnsiToWin32(mockStdout)
            self.assertFalse(stream.strip)

    def testWriteStripsAnsi(self):
        mockStdout = Mock()
        stream = AnsiToWin32(mockStdout)
        stream.wrapped = Mock()
        stream.write_and_convert = Mock()
        stream.strip = True
        stream.write('abc')
        self.assertFalse(stream.wrapped.write.called)
        self.assertEqual(stream.write_and_convert.call_args, (('abc',), {}))

    def testWriteDoesNotStripAnsi(self):
        mockStdout = Mock()
        stream = AnsiToWin32(mockStdout)
        stream.wrapped = Mock()
        stream.write_and_convert = Mock()
        stream.strip = False
        stream.convert = False
        stream.write('abc')
        self.assertFalse(stream.write_and_convert.called)
        self.assertEqual(stream.wrapped.write.call_args, (('abc',), {}))

    def assert_autoresets(self, convert, autoreset=True):
        stream = AnsiToWin32(Mock())
        stream.convert = convert
        stream.reset_all = Mock()
        stream.autoreset = autoreset
        stream.winterm = Mock()
        stream.write('abc')
        self.assertEqual(stream.reset_all.called, autoreset)

    def testWriteAutoresets(self):
        self.assert_autoresets(convert=True)
        self.assert_autoresets(convert=False)
        self.assert_autoresets(convert=True, autoreset=False)
        self.assert_autoresets(convert=False, autoreset=False)

    def testWriteAndConvertWritesPlainText(self):
        stream = AnsiToWin32(Mock())
        stream.write_and_convert('abc')
        self.assertEqual(stream.wrapped.write.call_args, (('abc',), {}))

    def testWriteAndConvertStripsAllValidAnsi(self):
        stream = AnsiToWin32(Mock())
        stream.call_win32 = Mock()
        data = ['abc\x1b[mdef', 'abc\x1b[0mdef', 'abc\x1b[2mdef', 'abc\x1b[02mdef', 'abc\x1b[002mdef', 'abc\x1b[40mdef', 'abc\x1b[040mdef', 'abc\x1b[0;1mdef', 'abc\x1b[40;50mdef', 'abc\x1b[50;30;40mdef', 'abc\x1b[Adef', 'abc\x1b[0Gdef', 'abc\x1b[1;20;128Hdef']
        for datum in data:
            stream.wrapped.write.reset_mock()
            stream.write_and_convert(datum)
            self.assertEqual([args[0] for args in stream.wrapped.write.call_args_list], [('abc',), ('def',)])

    def testWriteAndConvertSkipsEmptySnippets(self):
        stream = AnsiToWin32(Mock())
        stream.call_win32 = Mock()
        stream.write_and_convert('\x1b[40m\x1b[41m')
        self.assertFalse(stream.wrapped.write.called)

    def testWriteAndConvertCallsWin32WithParamsAndCommand(self):
        stream = AnsiToWin32(Mock())
        stream.convert = True
        stream.call_win32 = Mock()
        stream.extract_params = Mock(return_value='params')
        data = {'abc\x1b[adef': ('a', 'params'), 'abc\x1b[;;bdef': ('b', 'params'), 'abc\x1b[0cdef': ('c', 'params'), 'abc\x1b[;;0;;Gdef': ('G', 'params'), 'abc\x1b[1;20;128Hdef': ('H', 'params')}
        for datum, expected in data.items():
            stream.call_win32.reset_mock()
            stream.write_and_convert(datum)
            self.assertEqual(stream.call_win32.call_args[0], expected)

    def test_reset_all_shouldnt_raise_on_closed_orig_stdout(self):
        stream = StringIO()
        converter = AnsiToWin32(stream)
        stream.close()
        converter.reset_all()

    def test_wrap_shouldnt_raise_on_closed_orig_stdout(self):
        stream = StringIO()
        stream.close()
        with patch('colorama.ansitowin32.os.name', 'nt'), patch('colorama.ansitowin32.winapi_test', lambda: True):
            converter = AnsiToWin32(stream)
        self.assertTrue(converter.strip)
        self.assertFalse(converter.convert)

    def test_wrap_shouldnt_raise_on_missing_closed_attr(self):
        with patch('colorama.ansitowin32.os.name', 'nt'), patch('colorama.ansitowin32.winapi_test', lambda: True):
            converter = AnsiToWin32(object())
        self.assertTrue(converter.strip)
        self.assertFalse(converter.convert)

    def testExtractParams(self):
        stream = AnsiToWin32(Mock())
        data = {'': (0,), ';;': (0,), '2': (2,), ';;002;;': (2,), '0;1': (0, 1), ';;003;;456;;': (3, 456), '11;22;33;44;55': (11, 22, 33, 44, 55)}
        for datum, expected in data.items():
            self.assertEqual(stream.extract_params('m', datum), expected)

    def testCallWin32UsesLookup(self):
        listener = Mock()
        stream = AnsiToWin32(listener)
        stream.win32_calls = {1: (lambda *_, **__: listener(11),), 2: (lambda *_, **__: listener(22),), 3: (lambda *_, **__: listener(33),)}
        stream.call_win32('m', (3, 1, 99, 2))
        self.assertEqual([a[0][0] for a in listener.call_args_list], [33, 11, 22])

    def test_osc_codes(self):
        mockStdout = Mock()
        stream = AnsiToWin32(mockStdout, convert=True)
        with patch('colorama.ansitowin32.winterm') as winterm:
            data = ['\x1b]0\x07', '\x1b]0;foo\x08', '\x1b]0;colorama_test_title\x07', '\x1b]1;colorama_test_title\x07', '\x1b]2;colorama_test_title\x07', '\x1b]' + ';' * 64 + '\x08']
            for code in data:
                stream.write(code)
            self.assertEqual(winterm.set_title.call_count, 2)

    def test_native_windows_ansi(self):
        with ExitStack() as stack:

            def p(a, b):
                stack.enter_context(patch(a, b, create=True))
            p('colorama.ansitowin32.os.name', 'nt')
            p('colorama.ansitowin32.winapi_test', lambda: True)
            p('colorama.win32.winapi_test', lambda: True)
            p('colorama.winterm.win32.windll', 'non-None')
            p('colorama.winterm.get_osfhandle', lambda _: 1234)
            p('colorama.winterm.win32.GetConsoleMode', lambda _: ENABLE_VIRTUAL_TERMINAL_PROCESSING)
            SetConsoleMode = Mock()
            p('colorama.winterm.win32.SetConsoleMode', SetConsoleMode)
            stdout = Mock()
            stdout.closed = False
            stdout.isatty.return_value = True
            stdout.fileno.return_value = 1
            stream = AnsiToWin32(stdout)
            SetConsoleMode.assert_called_with(1234, ENABLE_VIRTUAL_TERMINAL_PROCESSING)
            self.assertFalse(stream.strip)
            self.assertFalse(stream.convert)
            self.assertFalse(stream.should_wrap())
            p('colorama.winterm.win32.GetConsoleMode', lambda _: 0)
            SetConsoleMode = Mock()
            p('colorama.winterm.win32.SetConsoleMode', SetConsoleMode)
            stream = AnsiToWin32(stdout)
            SetConsoleMode.assert_called_with(1234, ENABLE_VIRTUAL_TERMINAL_PROCESSING)
            self.assertTrue(stream.strip)
            self.assertTrue(stream.convert)
            self.assertTrue(stream.should_wrap())