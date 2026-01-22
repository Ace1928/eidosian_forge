from __future__ import unicode_literals
import unittest
from cmakelang.format import __main__
from cmakelang import markup
class MockMarkupConfig(object):

    def __init__(self, fence_pattern, ruler_pattern):
        self.fence_pattern = fence_pattern
        self.ruler_pattern = ruler_pattern