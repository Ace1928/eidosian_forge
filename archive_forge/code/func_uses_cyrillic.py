from typing import Optional
from docutils.writers.latex2e import Babel
def uses_cyrillic(self) -> bool:
    return self.language in self.cyrillic_languages