class IfParser:
    error_class = ValueError

    def __init__(self, tokens):
        num_tokens = len(tokens)
        mapped_tokens = []
        i = 0
        while i < num_tokens:
            token = tokens[i]
            if token == 'is' and i + 1 < num_tokens and (tokens[i + 1] == 'not'):
                token = 'is not'
                i += 1
            elif token == 'not' and i + 1 < num_tokens and (tokens[i + 1] == 'in'):
                token = 'not in'
                i += 1
            mapped_tokens.append(self.translate_token(token))
            i += 1
        self.tokens = mapped_tokens
        self.pos = 0
        self.current_token = self.next_token()

    def translate_token(self, token):
        try:
            op = OPERATORS[token]
        except (KeyError, TypeError):
            return self.create_var(token)
        else:
            return op()

    def next_token(self):
        if self.pos >= len(self.tokens):
            return EndToken
        else:
            retval = self.tokens[self.pos]
            self.pos += 1
            return retval

    def parse(self):
        retval = self.expression()
        if self.current_token is not EndToken:
            raise self.error_class("Unused '%s' at end of if expression." % self.current_token.display())
        return retval

    def expression(self, rbp=0):
        t = self.current_token
        self.current_token = self.next_token()
        left = t.nud(self)
        while rbp < self.current_token.lbp:
            t = self.current_token
            self.current_token = self.next_token()
            left = t.led(left, self)
        return left

    def create_var(self, value):
        return Literal(value)