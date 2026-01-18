from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import formatting
from fire import testutils
def test_wrap_one_item(self):
    lines = formatting.WrappedJoin(['rice'])
    self.assertEqual(['rice'], lines)