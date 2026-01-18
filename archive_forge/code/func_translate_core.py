import os
import re
def translate_core(self, pattern):
    """
        Given a glob pattern, produce a regex that matches it.

        >>> t = Translator()
        >>> t.translate_core('*.txt').replace('\\\\\\\\', '')
        '[^/]*\\\\.txt'
        >>> t.translate_core('a?txt')
        'a[^/]txt'
        >>> t.translate_core('**/*').replace('\\\\\\\\', '')
        '.*/[^/][^/]*'
        """
    self.restrict_rglob(pattern)
    return ''.join(map(self.replace, separate(self.star_not_empty(pattern))))