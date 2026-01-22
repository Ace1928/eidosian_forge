import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
def _encode_relation(self, name):
    """(INTERNAL) Decodes a relation line.

        The relation declaration is a line with the format ``@RELATION
        <relation-name>``, where ``relation-name`` is a string.

        :param name: a string.
        :return: a string with the encoded relation declaration.
        """
    for char in ' %{},':
        if char in name:
            name = '"%s"' % name
            break
    return '%s %s' % (_TK_RELATION, name)