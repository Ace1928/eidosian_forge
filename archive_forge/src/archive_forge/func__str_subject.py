@property
def _str_subject(self):
    target = self.target
    if target is self._NOT_GIVEN:
        return 'An object'
    return 'The object {!r}'.format(target)