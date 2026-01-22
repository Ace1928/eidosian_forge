import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
def _decode_comment(self, s):
    """(INTERNAL) Decodes a comment line.

        Comments are single line strings starting, obligatorily, with the ``%``
        character, and can have any symbol, including whitespaces or special
        characters.

        This method must receive a normalized string, i.e., a string without
        padding, including the "\r
" characters.

        :param s: a normalized string.
        :return: a string with the decoded comment.
        """
    res = re.sub('^\\%( )?', '', s)
    return res