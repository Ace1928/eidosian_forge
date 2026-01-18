import math
from typing import NamedTuple
from dataclasses import dataclass
def toPS(self):
    """Return a PostScript representation

        :Example:

                >>> t = Identity.scale(2, 3).translate(4, 5)
                >>> t.toPS()
                '[2 0 0 3 8 15]'
                >>>
        """
    return '[%s %s %s %s %s %s]' % self