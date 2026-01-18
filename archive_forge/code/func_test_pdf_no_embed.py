import os
from base64 import b64decode, b64encode
from io import BytesIO, StringIO
from pathlib import Path
import pytest
from requests.exceptions import MissingSchema
from panel.pane import (
from panel.pane.markup import escape
def test_pdf_no_embed(document, comm):
    url = 'https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf'
    pdf_pane = PDF(url, embed=False)
    model = pdf_pane.get_root(document, comm)
    assert model.text.startswith(f'&lt;embed src=&quot;{url}')