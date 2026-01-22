from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import sys
from antlr3 import runtime_version, runtime_version_str
from antlr3.compat import set, frozenset, reversed
from antlr3.constants import DEFAULT_CHANNEL, HIDDEN_CHANNEL, EOF, \
from antlr3.exceptions import RecognitionException, MismatchedTokenException, \
from antlr3.tokens import CommonToken, EOF_TOKEN, SKIP_TOKEN
import six
from six import unichr
class TokenSource(object):
    """
    @brief Abstract baseclass for token producers.

    A source of tokens must provide a sequence of tokens via nextToken()
    and also must reveal it's source of characters; CommonToken's text is
    computed from a CharStream; it only store indices into the char stream.

    Errors from the lexer are never passed to the parser.  Either you want
    to keep going or you do not upon token recognition error.  If you do not
    want to continue lexing then you do not want to continue parsing.  Just
    throw an exception not under RecognitionException and Java will naturally
    toss you all the way out of the recognizers.  If you want to continue
    lexing then you should not throw an exception to the parser--it has already
    requested a token.  Keep lexing until you get a valid one.  Just report
    errors and keep going, looking for a valid token.
    """

    def nextToken(self):
        """Return a Token object from your input stream (usually a CharStream).

        Do not fail/return upon lexing error; keep chewing on the characters
        until you get a good one; errors are not passed through to the parser.
        """
        raise NotImplementedError

    def __iter__(self):
        """The TokenSource is an interator.

        The iteration will not include the final EOF token, see also the note
        for the next() method.

        """
        return self

    def next(self):
        """Return next token or raise StopIteration.

        Note that this will raise StopIteration when hitting the EOF token,
        so EOF will not be part of the iteration.

        """
        token = self.nextToken()
        if token is None or token.type == EOF:
            raise StopIteration
        return token