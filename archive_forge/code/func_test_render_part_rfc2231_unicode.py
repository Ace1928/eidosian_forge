import pytest
from urllib3.fields import RequestField, format_header_param_rfc2231, guess_content_type
from urllib3.packages.six import u
def test_render_part_rfc2231_unicode(self):
    field = RequestField('somename', 'data', header_formatter=format_header_param_rfc2231)
    param = field._render_part('filename', u('näme'))
    assert param == "filename*=utf-8''n%C3%A4me"