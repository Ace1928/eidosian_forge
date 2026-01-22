from nltk.tokenize.api import StringTokenizer, TokenizerI
from nltk.tokenize.util import regexp_span_tokenize, string_span_tokenize
class CharTokenizer(StringTokenizer):
    """Tokenize a string into individual characters.  If this functionality
    is ever required directly, use ``for char in string``.
    """

    def tokenize(self, s):
        return list(s)

    def span_tokenize(self, s):
        yield from enumerate(range(1, len(s) + 1))