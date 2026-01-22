from __future__ import print_function
import sys
import textwrap
import pytest
from pathlib import Path
class TestNewAPI:

    def test_duplicate_keys_00(self):
        from srsly.ruamel_yaml import YAML
        from srsly.ruamel_yaml.constructor import DuplicateKeyError
        yaml = YAML()
        with pytest.raises(DuplicateKeyError):
            yaml.load('{a: 1, a: 2}')

    def test_duplicate_keys_01(self):
        from srsly.ruamel_yaml import YAML
        from srsly.ruamel_yaml.constructor import DuplicateKeyError
        yaml = YAML(typ='safe', pure=True)
        with pytest.raises(DuplicateKeyError):
            yaml.load('{a: 1, a: 2}')

    def test_duplicate_keys_02(self):
        from srsly.ruamel_yaml import YAML
        from srsly.ruamel_yaml.constructor import DuplicateKeyError
        yaml = YAML(typ='safe')
        with pytest.raises(DuplicateKeyError):
            yaml.load('{a: 1, a: 2}')

    def test_issue_135(self):
        from srsly.ruamel_yaml import YAML
        data = {'a': 1, 'b': 2}
        yaml = YAML(typ='safe')
        yaml.dump(data, sys.stdout)

    def test_issue_135_temporary_workaround(self):
        from srsly.ruamel_yaml import YAML
        data = {'a': 1, 'b': 2}
        yaml = YAML(typ='safe', pure=True)
        yaml.dump(data, sys.stdout)