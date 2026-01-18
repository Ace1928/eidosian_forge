def translate_token(self, token):
    try:
        op = OPERATORS[token]
    except (KeyError, TypeError):
        return self.create_var(token)
    else:
        return op()