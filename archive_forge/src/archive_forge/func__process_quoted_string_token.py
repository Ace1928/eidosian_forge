from collections import namedtuple
import re
import textwrap
import warnings
@classmethod
def _process_quoted_string_token(cls, token):
    """
        Return unescaped and unquoted value from quoted token.
        """
    return re.sub('\\\\(?![\\\\])', '', token[1:-1]).replace('\\\\', '\\')