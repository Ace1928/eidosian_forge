import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_xml_quoting_handles_angle_brackets(self):
    assert self.sub.substitute_xml('foo<bar>') == 'foo&lt;bar&gt;'