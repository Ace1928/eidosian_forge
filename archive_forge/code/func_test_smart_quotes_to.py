import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
@pytest.mark.parametrize('smart_quotes_to,expect_converted', [(None, '‘’“”'), ('xml', '&#x2018;&#x2019;&#x201C;&#x201D;'), ('html', '&lsquo;&rsquo;&ldquo;&rdquo;'), ('ascii', "''" + '""')])
def test_smart_quotes_to(self, smart_quotes_to, expect_converted):
    """Verify the functionality of the smart_quotes_to argument
        to the UnicodeDammit constructor."""
    markup = b'<foo>\x91\x92\x93\x94</foo>'
    converted = UnicodeDammit(markup, known_definite_encodings=['windows-1252'], smart_quotes_to=smart_quotes_to).unicode_markup
    assert converted == '<foo>{}</foo>'.format(expect_converted)