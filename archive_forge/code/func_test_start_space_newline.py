from .roundtrip import dedent
def test_start_space_newline(self):
    x = dedent('   \n        123\n        ')
    assert x == '123\n'