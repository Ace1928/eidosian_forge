from unittest import TestCase
from holoviews.element.comparison import Comparison as HvComparison
from .geo import Image, ImageStack, Points, LineContours, FilledContours, WindBarbs
class ComparisonTestCase(Comparison, TestCase):
    """
    Class to integrate the Comparison class with unittest.TestCase.
    """

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)
        registry = Comparison.register()
        for k, v in registry.items():
            self.addTypeEqualityFunc(k, v)