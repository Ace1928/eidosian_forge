import doctest
import os
import shutil
import tempfile
import unittest
from genshi.core import TEXT
from genshi.template.loader import TemplateLoader
from genshi.template.markup import MarkupTemplate
def test_relative_include_without_loader(self):
    file1 = open(os.path.join(self.dirname, 'tmpl1.html'), 'w')
    try:
        file1.write('<div>Included</div>')
    finally:
        file1.close()
    file2 = open(os.path.join(self.dirname, 'tmpl2.html'), 'w')
    try:
        file2.write('<html xmlns:xi="http://www.w3.org/2001/XInclude">\n              <xi:include href="tmpl1.html" />\n            </html>')
    finally:
        file2.close()
    tmpl = MarkupTemplate('<html xmlns:xi="http://www.w3.org/2001/XInclude">\n              <xi:include href="tmpl1.html" />\n            </html>', os.path.join(self.dirname, 'tmpl2.html'), 'tmpl2.html')
    self.assertEqual('<html>\n              <div>Included</div>\n            </html>', tmpl.generate().render(encoding=None))