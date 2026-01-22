import re
from ..core.inputscanner import InputScanner
from ..core.token import Token
from ..core.tokenstream import TokenStream
from ..core.pattern import Pattern
from ..core.whitespacepattern import WhitespacePattern
def __get_next_token_with_comments(self, previous, open_token):
    current = self._get_next_token(previous, open_token)
    if self._is_comment(current):
        comments = TokenStream()
        while self._is_comment(current):
            comments.add(current)
            current = self._get_next_token(previous, open_token)
        if not comments.isEmpty():
            current.comments_before = comments
            comments = TokenStream()
    current.parent = open_token
    current.previous = previous
    previous.next = current
    return current