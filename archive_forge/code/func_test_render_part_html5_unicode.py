import pytest
from urllib3.fields import RequestField, format_header_param_rfc2231, guess_content_type
from urllib3.packages.six import u
def test_render_part_html5_unicode(self):
    field = RequestField('somename', 'data')
    param = field._render_part('filename', u('näme'))
    assert param == u('filename="näme"')