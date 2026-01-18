import doctest
import os
import shutil
import tempfile
import unittest
from genshi.core import TEXT
from genshi.template.loader import TemplateLoader
from genshi.template.markup import MarkupTemplate

        Test prefix delegation with the following layout:
        
        templates/foo.html
        sub1/templates/tmpl1.html
        sub1/templates/tmpl2.html
        sub1/templates/bar/tmpl3.html
        
        Where sub1 is a prefix, and tmpl1.html includes all the others.
        