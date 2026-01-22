import unittest
from pkgutil import find_loader
class SVGDotOutputTest(DocumentBaseTestCase, AllTestsBase):
    """
        One-way output SVG with prov.dot to exercise its code
        """
    MIN_SVG_SIZE = 850

    def do_tests(self, prov_doc, msg=None):
        dot = prov_to_dot(prov_doc)
        svg_content = dot.create(format='svg', encoding='utf-8')
        self.assertGreater(len(svg_content), self.MIN_SVG_SIZE, 'The size of the generated SVG content should be greater than %d bytes' % self.MIN_SVG_SIZE)