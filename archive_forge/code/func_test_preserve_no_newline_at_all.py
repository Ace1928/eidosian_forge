from .roundtrip import dedent
def test_preserve_no_newline_at_all(self):
    x = dedent('        123')
    assert x == '123'