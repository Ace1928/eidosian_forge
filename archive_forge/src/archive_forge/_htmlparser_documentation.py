from html.parser import HTMLParser
import sys
import warnings
from bs4.element import (
from bs4.dammit import EntitySubstitution, UnicodeDammit
from bs4.builder import (
Run some incoming markup through some parsing process,
        populating the `BeautifulSoup` object in self.soup.
        