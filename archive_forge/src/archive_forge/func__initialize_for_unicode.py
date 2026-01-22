import unicodedata
import enchant.tokenize
def _initialize_for_unicode(self):
    self._consume_alpha = self._consume_alpha_u
    if self._valid_chars is None:
        self._valid_chars = ("'",)