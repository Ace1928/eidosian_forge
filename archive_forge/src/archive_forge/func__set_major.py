def _set_major(self, s):
    s = s.lower()
    if not is_token(s):
        raise ValueError('Major media type contains an invalid character')
    self._major = s