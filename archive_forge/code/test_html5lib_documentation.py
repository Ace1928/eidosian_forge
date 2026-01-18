import pytest
import warnings
from bs4 import BeautifulSoup
from bs4.element import SoupStrainer
from . import (

        Test that empty comment does not break structure.

        https://bugs.launchpad.net/beautifulsoup/+bug/1806598
        