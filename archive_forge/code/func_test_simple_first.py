from unittest import TestCase
import simplejson as json
from operator import itemgetter
def test_simple_first(self):
    a = {'a': 1, 'c': 5, 'jack': 'jill', 'pick': 'axe', 'array': [1, 5, 6, 9], 'tuple': (83, 12, 3), 'crate': 'dog', 'zeak': 'oh'}
    self.assertEqual('{"a": 1, "c": 5, "crate": "dog", "jack": "jill", "pick": "axe", "zeak": "oh", "array": [1, 5, 6, 9], "tuple": [83, 12, 3]}', json.dumps(a, item_sort_key=json.simple_first))