import os
from base64 import b64decode, b64encode
from io import BytesIO, StringIO
from pathlib import Path
import pytest
from requests.exceptions import MissingSchema
from panel.pane import (
from panel.pane.markup import escape
def test_pdf_local_file(document, comm):
    path = Path(__file__).parent.parent / 'test_data' / 'sample.pdf'
    pdf_pane = PDF(object=path)
    try:
        model = pdf_pane.get_root(document, comm)
    except MissingSchema:
        return
    assert model.text.startswith('&lt;embed src=&quot;data:application/pdf;base64,JVBER')
    assert model.text.endswith('==#page=1&quot; width=&#x27;100%&#x27; height=&#x27;100%&#x27; type=&quot;application/pdf&quot;&gt;')