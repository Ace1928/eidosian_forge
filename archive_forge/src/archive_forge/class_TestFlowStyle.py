from __future__ import print_function
import sys
import textwrap
import pytest
from pathlib import Path
class TestFlowStyle:

    def test_flow_style(self, capsys):
        from srsly.ruamel_yaml import YAML
        yaml = YAML()
        yaml.default_flow_style = None
        data = yaml.map()
        data['b'] = 1
        data['a'] = [[1, 2], [3, 4]]
        yaml.dump(data, sys.stdout)
        out, err = capsys.readouterr()
        assert out == 'b: 1\na:\n- [1, 2]\n- [3, 4]\n'