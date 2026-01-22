import unicodedata
def _is_unicat(self, var, cat):
    return unicodedata.category(var) == cat