import rpy2.rlike.indexing as rli
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
def __imul__(self, y):
    restags = self.__tags.__imul__(y)
    resitems = super(TaggedList, self).__imul__(y)
    return self