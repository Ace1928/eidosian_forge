class TokenMismatch(LockBroken):
    _fmt = 'The lock token %(given_token)r does not match lock token %(lock_token)r.'
    internal_error = True

    def __init__(self, given_token, lock_token):
        self.given_token = given_token
        self.lock_token = lock_token