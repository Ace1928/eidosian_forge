from html5lib.treebuilders import _base, etree as etree_builders
from lxml import html, etree

Legacy module - don't use in new code!

html5lib now has its own proper implementation.

This module implements a tree builder for html5lib that generates lxml
html element trees.  This module uses camelCase as it follows the
html5lib style guide.
