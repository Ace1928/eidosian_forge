import doctest
import os
import shutil
import tempfile
import unittest
from genshi.core import TEXT
from genshi.template.loader import TemplateLoader
from genshi.template.markup import MarkupTemplate
def test_prefix_delegation_to_directories_with_subdirs(self):
    """
        Test prefix delegation with the following layout:
        
        templates/foo.html
        sub1/templates/tmpl1.html
        sub1/templates/tmpl2.html
        sub1/templates/bar/tmpl3.html
        
        Where sub1 is a prefix, and tmpl1.html includes all the others.
        """
    dir1 = os.path.join(self.dirname, 'templates')
    os.mkdir(dir1)
    file1 = open(os.path.join(dir1, 'foo.html'), 'w')
    try:
        file1.write('<div>Included foo</div>')
    finally:
        file1.close()
    dir2 = os.path.join(self.dirname, 'sub1', 'templates')
    os.makedirs(dir2)
    file2 = open(os.path.join(dir2, 'tmpl1.html'), 'w')
    try:
        file2.write('<html xmlns:xi="http://www.w3.org/2001/XInclude">\n              <xi:include href="../foo.html" /> from sub1\n              <xi:include href="tmpl2.html" /> from sub1\n              <xi:include href="bar/tmpl3.html" /> from sub1\n            </html>')
    finally:
        file2.close()
    file3 = open(os.path.join(dir2, 'tmpl2.html'), 'w')
    try:
        file3.write('<div>tmpl2</div>')
    finally:
        file3.close()
    dir3 = os.path.join(self.dirname, 'sub1', 'templates', 'bar')
    os.makedirs(dir3)
    file4 = open(os.path.join(dir3, 'tmpl3.html'), 'w')
    try:
        file4.write('<div>bar/tmpl3</div>')
    finally:
        file4.close()
    loader = TemplateLoader([dir1, TemplateLoader.prefixed(sub1=os.path.join(dir2), sub2=os.path.join(dir3))])
    tmpl = loader.load('sub1/tmpl1.html')
    self.assertEqual('<html>\n              <div>Included foo</div> from sub1\n              <div>tmpl2</div> from sub1\n              <div>bar/tmpl3</div> from sub1\n            </html>', tmpl.generate().render(encoding=None))