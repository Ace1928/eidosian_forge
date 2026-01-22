from io import StringIO, TextIOWrapper
from email.feedparser import FeedParser, BytesFeedParser
from email._policybase import compat32
class BytesParser:

    def __init__(self, *args, **kw):
        """Parser of binary RFC 2822 and MIME email messages.

        Creates an in-memory object tree representing the email message, which
        can then be manipulated and turned over to a Generator to return the
        textual representation of the message.

        The input must be formatted as a block of RFC 2822 headers and header
        continuation lines, optionally preceded by a `Unix-from' header.  The
        header block is terminated either by the end of the input or by a
        blank line.

        _class is the class to instantiate for new message objects when they
        must be created.  This class must have a constructor that can take
        zero arguments.  Default is Message.Message.
        """
        self.parser = Parser(*args, **kw)

    def parse(self, fp, headersonly=False):
        """Create a message structure from the data in a binary file.

        Reads all the data from the file and returns the root of the message
        structure.  Optional headersonly is a flag specifying whether to stop
        parsing after reading the headers or not.  The default is False,
        meaning it parses the entire contents of the file.
        """
        fp = TextIOWrapper(fp, encoding='ascii', errors='surrogateescape')
        try:
            return self.parser.parse(fp, headersonly)
        finally:
            fp.detach()

    def parsebytes(self, text, headersonly=False):
        """Create a message structure from a byte string.

        Returns the root of the message structure.  Optional headersonly is a
        flag specifying whether to stop parsing after reading the headers or
        not.  The default is False, meaning it parses the entire contents of
        the file.
        """
        text = text.decode('ASCII', errors='surrogateescape')
        return self.parser.parsestr(text, headersonly)