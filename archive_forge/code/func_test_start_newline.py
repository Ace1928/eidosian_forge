from .roundtrip import dedent
def test_start_newline(self):
    x = dedent('\n        123\n          456\n        ')
    assert x == '123\n  456\n'