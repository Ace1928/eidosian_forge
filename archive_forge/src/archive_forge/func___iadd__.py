import rpy2.rlike.indexing as rli
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
def __iadd__(self, y):
    super(TaggedList, self).__iadd__(y)
    if isinstance(y, TaggedList):
        self.__tags.__iadd__(y.tags)
    else:
        self.__tags.__iadd__([None] * len(y))
    return self