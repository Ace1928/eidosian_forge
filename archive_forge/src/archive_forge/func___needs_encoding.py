import re
def __needs_encoding(self, s):
    """
        Get whether string I{s} contains special characters.

        @param s: A string to check.
        @type s: str
        @return: True if needs encoding.
        @rtype: boolean

        """
    if isinstance(s, str):
        for c in self.special:
            if c in s:
                return True