import sys
from typing import NoReturn
from twisted.trial.unittest import TestCase
from twisted.web.template import CDATA, CharRef, Comment, Flattenable, Tag
def test_renderAttribute(self) -> None:
    """
        Setting an attribute named C{render} will change the C{render} instance
        variable instead of adding an attribute.
        """
    tag = proto(render='myRenderer')
    self.assertEqual(tag.render, 'myRenderer')
    self.assertEqual(tag.attributes, {})