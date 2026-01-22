from nltk.tokenize.api import StringTokenizer, TokenizerI
from nltk.tokenize.util import regexp_span_tokenize, string_span_tokenize
class TabTokenizer(StringTokenizer):
    """Tokenize a string use the tab character as a delimiter,
    the same as ``s.split('\\t')``.

        >>> from nltk.tokenize import TabTokenizer
        >>> TabTokenizer().tokenize('a\\tb c\\n\\t d')
        ['a', 'b c\\n', ' d']
    """
    _string = '\t'