import doctest
import os
import shutil
import tempfile
import unittest
from genshi.core import TEXT
from genshi.template.loader import TemplateLoader
from genshi.template.markup import MarkupTemplate
def test_relative_include_without_search_path_nested(self):
    file1 = open(os.path.join(self.dirname, 'tmpl1.html'), 'w')
    try:
        file1.write('<div>Included</div>')
    finally:
        file1.close()
    file2 = open(os.path.join(self.dirname, 'tmpl2.html'), 'w')
    try:
        file2.write('<div xmlns:xi="http://www.w3.org/2001/XInclude">\n              <xi:include href="tmpl1.html" />\n            </div>')
    finally:
        file2.close()
    file3 = open(os.path.join(self.dirname, 'tmpl3.html'), 'w')
    try:
        file3.write('<html xmlns:xi="http://www.w3.org/2001/XInclude">\n              <xi:include href="tmpl2.html" />\n            </html>')
    finally:
        file3.close()
    loader = TemplateLoader()
    tmpl = loader.load(os.path.join(self.dirname, 'tmpl3.html'))
    self.assertEqual('<html>\n              <div>\n              <div>Included</div>\n            </div>\n            </html>', tmpl.generate().render(encoding=None))