import io
import os.path
from IPython.utils import openpy
def test_source_to_unicode():
    with io.open(nonascii_path, 'rb') as f:
        source_bytes = f.read()
    assert openpy.source_to_unicode(source_bytes, skip_encoding_cookie=False).splitlines() == source_bytes.decode('iso-8859-5').splitlines()
    source_no_cookie = openpy.source_to_unicode(source_bytes, skip_encoding_cookie=True)
    assert 'coding: iso-8859-5' not in source_no_cookie