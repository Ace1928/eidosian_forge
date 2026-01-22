def _dict_match(self, partial, real):
    result = True
    try:
        for key, value in partial.items():
            if isinstance(value, dict):
                result = self._dict_match(value, real[key])
            else:
                assert real[key] == value
                result = True
    except (AssertionError, KeyError):
        result = False
    return result