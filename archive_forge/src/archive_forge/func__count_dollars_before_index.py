import re
import textwrap
def _count_dollars_before_index(self, s, i):
    """Returns the number of '$' characters right in front of s[i]."""
    dollar_count = 0
    dollar_index = i - 1
    while dollar_index > 0 and s[dollar_index] == '$':
        dollar_count += 1
        dollar_index -= 1
    return dollar_count