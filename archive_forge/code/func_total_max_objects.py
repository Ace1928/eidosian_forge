from collections import defaultdict, Counter
from typing import Dict, Generator, List, Optional, TypeVar
@property
def total_max_objects(self):
    return sum(self._max_num_objects.values())