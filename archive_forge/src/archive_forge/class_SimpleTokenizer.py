import regex
from .tokenizer import Tokens, Tokenizer
from parlai.utils.logging import logger
class SimpleTokenizer(Tokenizer):
    ALPHA_NUM = '[\\p{L}\\p{N}\\p{M}]+'
    NON_WS = '[^\\p{Z}\\p{C}]'

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile('(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS), flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE)
        if len(kwargs.get('annotators', {})) > 0:
            logger.warning('%s only tokenizes! Skipping annotators: %s' % (type(self).__name__, kwargs.get('annotators')))
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            token = matches[i].group()
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]
            data.append((token, text[start_ws:end_ws], span))
        return Tokens(data, self.annotators)