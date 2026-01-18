import array
import warnings
import enchant
from enchant.errors import (
from enchant.tokenize import get_tokenizer
from enchant.utils import get_default_language
def trailing_context(self, chars):
    """Get <chars> characters of trailing context.

        This method returns up to <chars> characters of trailing
        context - the text that occurs in the string immediately
        after the current erroneous word.
        """
    start = self.wordpos + len(self.word)
    end = min(start + chars, len(self._text))
    context = self._text[start:end]
    return self._array_to_string(context)