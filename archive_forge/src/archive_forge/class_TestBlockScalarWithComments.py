import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
class TestBlockScalarWithComments:

    def test_scalar_with_comments(self):
        import srsly.ruamel_yaml
        for x in ['', '\n', '\n# Another comment\n', '\n\n', '\n\n# abc\n#xyz\n', '\n\n# abc\n#xyz\n', '# abc\n\n#xyz\n', '\n\n  # abc\n  #xyz\n']:
            commented_line = test_block_scalar_commented_line_template.format(x)
            data = srsly.ruamel_yaml.round_trip_load(commented_line)
            assert srsly.ruamel_yaml.round_trip_dump(data) == commented_line