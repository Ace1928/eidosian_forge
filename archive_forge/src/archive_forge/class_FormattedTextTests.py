import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
class FormattedTextTests(IRCTestCase):
    """
    Tests for parsing and assembling formatted IRC text.
    """

    def assertAssembledEqually(self, text, expectedFormatted):
        """
        Assert that C{text} is parsed and assembled to the same value as what
        C{expectedFormatted} is assembled to. This provides a way to ignore
        meaningless differences in the formatting structure that would be
        difficult to detect without rendering the structures.
        """
        formatted = irc.parseFormattedText(text)
        self.assertAssemblesTo(formatted, expectedFormatted)

    def assertAssemblesTo(self, formatted, expectedFormatted):
        """
        Assert that C{formatted} and C{expectedFormatted} assemble to the same
        value.
        """
        text = irc.assembleFormattedText(formatted)
        expectedText = irc.assembleFormattedText(expectedFormatted)
        self.assertEqual(irc.assembleFormattedText(formatted), expectedText, '%r (%r) is not equivalent to %r (%r)' % (text, formatted, expectedText, expectedFormatted))

    def test_parseEmpty(self):
        """
        An empty string parses to a I{normal} attribute with no text.
        """
        self.assertAssembledEqually('', A.normal)

    def test_assembleEmpty(self):
        """
        An attribute with no text assembles to the empty string. An attribute
        whose text is the empty string assembles to two control codes: C{off}
        and that of the attribute.
        """
        self.assertEqual(irc.assembleFormattedText(A.normal), '')
        self.assertEqual(irc.assembleFormattedText(A.bold['']), '\x0f\x02')

    def test_assembleNormal(self):
        """
        A I{normal} string assembles to a string prefixed with the I{off}
        control code.
        """
        self.assertEqual(irc.assembleFormattedText(A.normal['hello']), '\x0fhello')

    def test_assembleBold(self):
        """
        A I{bold} string assembles to a string prefixed with the I{off} and
        I{bold} control codes.
        """
        self.assertEqual(irc.assembleFormattedText(A.bold['hello']), '\x0f\x02hello')

    def test_assembleUnderline(self):
        """
        An I{underline} string assembles to a string prefixed with the I{off}
        and I{underline} control codes.
        """
        self.assertEqual(irc.assembleFormattedText(A.underline['hello']), '\x0f\x1fhello')

    def test_assembleReverseVideo(self):
        """
        A I{reverse video} string assembles to a string prefixed with the I{off}
        and I{reverse video} control codes.
        """
        self.assertEqual(irc.assembleFormattedText(A.reverseVideo['hello']), '\x0f\x16hello')

    def test_assembleForegroundColor(self):
        """
        A I{foreground color} string assembles to a string prefixed with the
        I{off} and I{color} (followed by the relevant foreground color code)
        control codes.
        """
        self.assertEqual(irc.assembleFormattedText(A.fg.blue['hello']), '\x0f\x0302hello')

    def test_assembleBackgroundColor(self):
        """
        A I{background color} string assembles to a string prefixed with the
        I{off} and I{color} (followed by a I{,} to indicate the absence of a
        foreground color, followed by the relevant background color code)
        control codes.
        """
        self.assertEqual(irc.assembleFormattedText(A.bg.blue['hello']), '\x0f\x03,02hello')

    def test_assembleColor(self):
        """
        A I{foreground} and I{background} color string assembles to a string
        prefixed with the I{off} and I{color} (followed by the relevant
        foreground color, I{,} and the relevant background color code) control
        codes.
        """
        self.assertEqual(irc.assembleFormattedText(A.fg.red[A.bg.blue['hello']]), '\x0f\x0305,02hello')

    def test_assembleNested(self):
        """
        Nested attributes retain the attributes of their parents.
        """
        self.assertEqual(irc.assembleFormattedText(A.bold['hello', A.underline[' world']]), '\x0f\x02hello\x0f\x02\x1f world')
        self.assertEqual(irc.assembleFormattedText(A.normal[A.fg.red[A.bg.green['hello'], ' world'], A.reverseVideo[' yay']]), '\x0f\x0305,03hello\x0f\x0305 world\x0f\x16 yay')

    def test_parseUnformattedText(self):
        """
        Parsing unformatted text results in text with attributes that
        constitute a no-op.
        """
        self.assertEqual(irc.parseFormattedText('hello'), A.normal['hello'])

    def test_colorFormatting(self):
        """
        Correctly formatted text with colors uses 2 digits to specify
        foreground and (optionally) background.
        """
        self.assertEqual(irc.parseFormattedText('\x0301yay\x03'), A.fg.black['yay'])
        self.assertEqual(irc.parseFormattedText('\x0301,02yay\x03'), A.fg.black[A.bg.blue['yay']])
        self.assertEqual(irc.parseFormattedText('\x0301yay\x0302yipee\x03'), A.fg.black['yay', A.fg.blue['yipee']])

    def test_weirdColorFormatting(self):
        """
        Formatted text with colors can use 1 digit for both foreground and
        background, as long as the text part does not begin with a digit.
        Foreground and background colors are only processed to a maximum of 2
        digits per component, anything else is treated as text. Color sequences
        must begin with a digit, otherwise processing falls back to unformatted
        text.
        """
        self.assertAssembledEqually('\x031kinda valid', A.fg.black['kinda valid'])
        self.assertAssembledEqually('\x03999,999kinda valid', A.fg.green['9,999kinda valid'])
        self.assertAssembledEqually('\x031,2kinda valid', A.fg.black[A.bg.blue['kinda valid']])
        self.assertAssembledEqually('\x031,999kinda valid', A.fg.black[A.bg.green['9kinda valid']])
        self.assertAssembledEqually('\x031,242 is a special number', A.fg.black[A.bg.yellow['2 is a special number']])
        self.assertAssembledEqually('\x03,02oops\x03', A.normal[',02oops'])
        self.assertAssembledEqually('\x03wrong', A.normal['wrong'])
        self.assertAssembledEqually('\x031,hello', A.fg.black['hello'])
        self.assertAssembledEqually('\x03\x03', A.normal)

    def test_clearColorFormatting(self):
        """
        An empty color format specifier clears foreground and background
        colors.
        """
        self.assertAssembledEqually('\x0301yay\x03reset', A.normal[A.fg.black['yay'], 'reset'])
        self.assertAssembledEqually('\x0301,02yay\x03reset', A.normal[A.fg.black[A.bg.blue['yay']], 'reset'])

    def test_resetFormatting(self):
        """
        A reset format specifier clears all formatting attributes.
        """
        self.assertAssembledEqually('\x02\x1fyay\x0freset', A.normal[A.bold[A.underline['yay']], 'reset'])
        self.assertAssembledEqually('\x0301yay\x0freset', A.normal[A.fg.black['yay'], 'reset'])
        self.assertAssembledEqually('\x0301,02yay\x0freset', A.normal[A.fg.black[A.bg.blue['yay']], 'reset'])

    def test_stripFormatting(self):
        """
        Strip formatting codes from formatted text, leaving only the text parts.
        """
        self.assertEqual(irc.stripFormatting(irc.assembleFormattedText(A.bold[A.underline[A.reverseVideo[A.fg.red[A.bg.green['hello']]], ' world']])), 'hello world')