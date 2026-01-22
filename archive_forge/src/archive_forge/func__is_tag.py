import re
import warnings
import array
from enchant.errors import TokenizerNotFoundError
def _is_tag(self, text, offset):
    if offset + 1 < len(text):
        if text[offset + 1].isalpha():
            return True
        if text[offset + 1] == '/':
            return True
    return False