from pdb import set_trace
import logging
import os
import pickle
import pytest
import sys
import tempfile
from bs4 import (
from bs4.builder import (
from bs4.element import (
from . import (
import warnings
def test_replacement_classes(self):

    class TagPlus(Tag):
        pass

    class StringPlus(NavigableString):
        pass

    class CommentPlus(Comment):
        pass
    soup = self.soup('<a><b>foo</b>bar</a><!--whee-->', element_classes={Tag: TagPlus, NavigableString: StringPlus, Comment: CommentPlus})
    assert all((isinstance(x, (TagPlus, StringPlus, CommentPlus)) for x in soup.recursiveChildGenerator()))