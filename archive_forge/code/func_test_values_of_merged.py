import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
def test_values_of_merged(self):
    from srsly.ruamel_yaml import YAML
    yaml = YAML()
    data = yaml.load(dedent(self.yaml_str))
    assert list(data[2].values()) == [1, 6, 'x2', 'x3', 'y4']