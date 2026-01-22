import re
from typing import List, Optional, Tuple
class _Backslash:

    def __init__(self, exit_state):
        self.exit_state = exit_state
        self.count = 1

    def process(self, next_char, context):
        if next_char == '\\':
            self.count += 1
            return self
        elif next_char in context.allowed_quote_chars:
            context.token.append('\\' * (self.count // 2))
            if self.count % 2 == 1:
                context.token.append(next_char)
            else:
                context.seq.pushback(next_char)
            self.count = 0
            return self.exit_state
        else:
            if self.count > 0:
                context.token.append('\\' * self.count)
                self.count = 0
            context.seq.pushback(next_char)
            return self.exit_state

    def finish(self, context):
        if self.count > 0:
            context.token.append('\\' * self.count)