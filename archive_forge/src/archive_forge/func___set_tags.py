import rpy2.rlike.indexing as rli
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
def __set_tags(self, tags):
    if len(tags) == len(self.__tags):
        self.__tags = tuple(tags)
    else:
        raise ValueError('The new list of tags should have the same length as the old one.')