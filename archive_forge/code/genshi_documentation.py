from __future__ import absolute_import, division, unicode_literals
from genshi.core import QName, Attrs
from genshi.core import START, END, TEXT, COMMENT, DOCTYPE
Convert a tree to a genshi tree

    :arg walker: the treewalker to use to walk the tree to convert it

    :returns: generator of genshi nodes

    