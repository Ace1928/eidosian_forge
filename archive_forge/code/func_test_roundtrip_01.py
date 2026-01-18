import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
def test_roundtrip_01(self):
    yaml_str = '\n        - &dotted.words.here[a, b]\n        - *dotted.words.here\n        '
    data = load(yaml_str)
    compare(data, yaml_str.replace('[', ' ['))