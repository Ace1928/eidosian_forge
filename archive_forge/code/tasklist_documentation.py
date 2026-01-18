import markdown
from functools import reduce
from markdown.treeprocessors import Treeprocessor
import xml.etree.ElementTree as etree

    An extension that supports GitHub task lists. Both ordered and unordered
    lists are supported and can be separately enabled. Nested lists are
    supported.

    Example::

       - [x] milk
       - [ ] eggs
       - [x] chocolate
       - [ ] if possible:
           1. [ ] solve world peace
           2. [ ] solve world hunger
    