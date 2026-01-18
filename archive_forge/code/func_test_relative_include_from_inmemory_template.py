import doctest
import os
import shutil
import tempfile
import unittest
from genshi.core import TEXT
from genshi.template.loader import TemplateLoader
from genshi.template.markup import MarkupTemplate
def test_relative_include_from_inmemory_template(self):
    file1 = open(os.path.join(self.dirname, 'tmpl1.html'), 'w')
    try:
        file1.write('<div>Included</div>')
    finally:
        file1.close()
    loader = TemplateLoader([self.dirname])
    tmpl2 = MarkupTemplate('<html xmlns:xi="http://www.w3.org/2001/XInclude">\n          <xi:include href="../tmpl1.html" />\n        </html>', filename='subdir/tmpl2.html', loader=loader)
    self.assertEqual('<html>\n          <div>Included</div>\n        </html>', tmpl2.generate().render(encoding=None))