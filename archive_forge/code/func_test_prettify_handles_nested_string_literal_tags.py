import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_prettify_handles_nested_string_literal_tags(self):
    markup = '<div><pre><code>some\n<script><pre>code</pre></script> for you \n</code></pre></div>'
    expect = '<div>\n <pre><code>some\n<script><pre>code</pre></script> for you \n</code></pre>\n</div>\n'
    soup = self.soup(markup)
    assert expect == soup.div.prettify()