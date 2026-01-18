import pytest
from urllib3.fields import RequestField, format_header_param_rfc2231, guess_content_type
from urllib3.packages.six import u
def test_render_parts(self):
    field = RequestField('somename', 'data')
    parts = field._render_parts({'name': 'value', 'filename': 'value'})
    assert 'name="value"' in parts
    assert 'filename="value"' in parts
    parts = field._render_parts([('name', 'value'), ('filename', 'value')])
    assert parts == 'name="value"; filename="value"'