from .roundtrip import dedent
class TestDedent:

    def test_start_newline(self):
        x = dedent('\n        123\n          456\n        ')
        assert x == '123\n  456\n'

    def test_start_space_newline(self):
        x = dedent('   \n        123\n        ')
        assert x == '123\n'

    def test_start_no_newline(self):
        x = dedent('        123\n          456\n        ')
        assert x == '123\n  456\n'

    def test_preserve_no_newline_at_end(self):
        x = dedent('\n        123')
        assert x == '123'

    def test_preserve_no_newline_at_all(self):
        x = dedent('        123')
        assert x == '123'

    def test_multiple_dedent(self):
        x = dedent(dedent('\n        123\n        '))
        assert x == '123\n'