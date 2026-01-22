import re
def _split_chunks(self, text):
    text = self._munge_whitespace(text)
    return self._split(text)