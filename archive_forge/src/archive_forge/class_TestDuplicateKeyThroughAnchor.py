import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
class TestDuplicateKeyThroughAnchor:

    def test_duplicate_key_00(self):
        from srsly.ruamel_yaml import version_info
        from srsly.ruamel_yaml import safe_load, round_trip_load
        from srsly.ruamel_yaml.constructor import DuplicateKeyFutureWarning, DuplicateKeyError
        s = dedent('        &anchor foo:\n            foo: bar\n            *anchor : duplicate key\n            baz: bat\n            *anchor : duplicate key\n        ')
        if version_info < (0, 15, 1):
            pass
        elif version_info < (0, 16, 0):
            with pytest.warns(DuplicateKeyFutureWarning):
                safe_load(s)
            with pytest.warns(DuplicateKeyFutureWarning):
                round_trip_load(s)
        else:
            with pytest.raises(DuplicateKeyError):
                safe_load(s)
            with pytest.raises(DuplicateKeyError):
                round_trip_load(s)

    def test_duplicate_key_01(self):
        from srsly.ruamel_yaml import version_info
        from srsly.ruamel_yaml.constructor import DuplicateKeyError
        s = dedent('        - &name-name\n          a: 1\n        - &help-name\n          b: 2\n        - <<: *name-name\n          <<: *help-name\n        ')
        if version_info < (0, 15, 1):
            pass
        else:
            with pytest.raises(DuplicateKeyError):
                yaml = YAML(typ='safe')
                yaml.load(s)
            with pytest.raises(DuplicateKeyError):
                yaml = YAML()
                yaml.load(s)