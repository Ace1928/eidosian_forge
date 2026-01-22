from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
from antlr3.constants import DEFAULT_CHANNEL, EOF
from antlr3.tokens import Token, EOF_TOKEN
import six
from six import StringIO
class TokenStream(IntStream):
    """

    @brief A stream of tokens accessing tokens from a TokenSource

    This is an abstract class that must be implemented by a subclass.

    """

    def LT(self, k):
        """
        Get Token at current input pointer + i ahead where i=1 is next Token.
        i<0 indicates tokens in the past.  So -1 is previous token and -2 is
        two tokens ago. LT(0) is undefined.  For i>=n, return Token.EOFToken.
        Return null for LT(0) and any index that results in an absolute address
        that is negative.
        """
        raise NotImplementedError

    def get(self, i):
        """
        Get a token at an absolute index i; 0..n-1.  This is really only
        needed for profiling and debugging and token stream rewriting.
        If you don't want to buffer up tokens, then this method makes no
        sense for you.  Naturally you can't use the rewrite stream feature.
        I believe DebugTokenStream can easily be altered to not use
        this method, removing the dependency.
        """
        raise NotImplementedError

    def getTokenSource(self):
        """
        Where is this stream pulling tokens from?  This is not the name, but
        the object that provides Token objects.
        """
        raise NotImplementedError

    def toString(self, start=None, stop=None):
        """
        Return the text of all tokens from start to stop, inclusive.
        If the stream does not buffer all the tokens then it can just
        return "" or null;  Users should not access $ruleLabel.text in
        an action of course in that case.

        Because the user is not required to use a token with an index stored
        in it, we must provide a means for two token objects themselves to
        indicate the start/end location.  Most often this will just delegate
        to the other toString(int,int).  This is also parallel with
        the TreeNodeStream.toString(Object,Object).
        """
        raise NotImplementedError