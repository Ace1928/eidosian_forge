import re
from typing import List, Optional, Tuple
class _Quotes:

    def __init__(self, quote_char, exit_state):
        self.quote_char = quote_char
        self.exit_state = exit_state

    def process(self, next_char, context):
        if next_char == '\\':
            return _Backslash(self)
        elif next_char == self.quote_char:
            context.token.append('')
            return self.exit_state
        else:
            context.token.append(next_char)
            return self