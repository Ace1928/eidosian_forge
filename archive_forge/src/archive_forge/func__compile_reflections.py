import random
import re
def _compile_reflections(self):
    sorted_refl = sorted(self._reflections, key=len, reverse=True)
    return re.compile('\\b({})\\b'.format('|'.join(map(re.escape, sorted_refl))), re.IGNORECASE)