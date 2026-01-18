import os
import pytest
from bs4 import (
@pytest.mark.parametrize('filename', ['clusterfuzz-testcase-minimized-bs4_fuzzer-5703933063462912', 'crash-ffbdfa8a2b26f13537b68d3794b0478a4090ee4a'])
def test_rejected_markup(self, filename):
    markup = self.__markup(filename)
    with pytest.raises(ParserRejectedMarkup):
        BeautifulSoup(markup, 'html.parser')