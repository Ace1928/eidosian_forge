import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
class TestCurrentDictKey(LineTestCase):

    def setUp(self):
        self.func = current_dict_key

    def test_simple(self):
        self.assertAccess('asdf|')
        self.assertAccess('asdf|')
        self.assertAccess('asdf[<>|')
        self.assertAccess('asdf[<>|]')
        self.assertAccess('object.dict[<abc|>')
        self.assertAccess('asdf|')
        self.assertAccess('asdf[<(>|]')
        self.assertAccess('asdf[<(1>|]')
        self.assertAccess('asdf[<(1,>|]')
        self.assertAccess('asdf[<(1,)>|]')
        self.assertAccess('asdf[<(1, >|]')
        self.assertAccess('asdf[<(1, 2)>|]')
        self.assertAccess("d[<'a>|")
        self.assertAccess("object.dict['a'bcd'], object.dict[<'abc>|")
        self.assertAccess("object.dict[<'a'bcd'>|], object.dict['abc")
        self.assertAccess('object.dict[<\'a\\\'\\\\\\"\\n\\\\\'>|')
        self.assertAccess('object.dict[<"abc\'>|')
        self.assertAccess("object.dict[<(1, 'apple', 2.134>|]")
        self.assertAccess("object.dict[<(1, 'apple', 2.134)>|]")
        self.assertAccess('object.dict[<-1000>|')
        self.assertAccess('object.dict[<-0.23948>|')
        self.assertAccess("object.dict[<'\U0001ffff>|")
        self.assertAccess('object.dict[<\'a\\\'\\\\\\"\\n\\\\\'>|]')
        self.assertAccess('object.dict[<\'a\\\'\\\\\\"\\n\\\\|[[]\'>')
        self.assertAccess('object.dict[<"a]bc[|]">]')
        self.assertAccess("object.dict[<'abcd[]>|")