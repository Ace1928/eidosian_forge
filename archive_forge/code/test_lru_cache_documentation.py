from dulwich import lru_cache
from dulwich.tests import TestCase
The cache is cleared in LRU order until small enough.