from __future__ import print_function
import pytest  # NOQA
from .roundtrip import YAML  # does an automatic dedent on load
class Test_RoundTripLiteral:

    def test_rt_root_literal_scalar_no_indent(self):
        yaml = YAML()
        yaml.explicit_start = True
        s = 'testing123'
        ys = '\n        --- |\n        {}\n        '
        ys = ys.format(s)
        d = yaml.load(ys)
        yaml.dump(d, compare=ys)

    def test_rt_root_literal_scalar_indent(self):
        yaml = YAML()
        yaml.explicit_start = True
        yaml.indent = 4
        s = 'testing123'
        ys = '\n        --- |\n            {}\n        '
        ys = ys.format(s)
        d = yaml.load(ys)
        yaml.dump(d, compare=ys)

    def test_rt_root_plain_scalar_no_indent(self):
        yaml = YAML()
        yaml.explicit_start = True
        yaml.indent = 0
        s = 'testing123'
        ys = '\n        ---\n        {}\n        '
        ys = ys.format(s)
        d = yaml.load(ys)
        yaml.dump(d, compare=ys)

    def test_rt_root_plain_scalar_expl_indent(self):
        yaml = YAML()
        yaml.explicit_start = True
        yaml.indent = 4
        s = 'testing123'
        ys = '\n        ---\n            {}\n        '
        ys = ys.format(s)
        d = yaml.load(ys)
        yaml.dump(d, compare=ys)

    def test_rt_root_sq_scalar_expl_indent(self):
        yaml = YAML()
        yaml.explicit_start = True
        yaml.indent = 4
        s = "'testing: 123'"
        ys = '\n        ---\n            {}\n        '
        ys = ys.format(s)
        d = yaml.load(ys)
        yaml.dump(d, compare=ys)

    def test_rt_root_dq_scalar_expl_indent(self):
        yaml = YAML()
        yaml.explicit_start = True
        yaml.indent = 0
        s = '"\'testing123"'
        ys = '\n        ---\n        {}\n        '
        ys = ys.format(s)
        d = yaml.load(ys)
        yaml.dump(d, compare=ys)

    def test_rt_root_literal_scalar_no_indent_no_eol(self):
        yaml = YAML()
        yaml.explicit_start = True
        s = 'testing123'
        ys = '\n        --- |-\n        {}\n        '
        ys = ys.format(s)
        d = yaml.load(ys)
        yaml.dump(d, compare=ys)

    def test_rt_non_root_literal_scalar(self):
        yaml = YAML()
        s = 'testing123'
        ys = '\n        - |\n          {}\n        '
        ys = ys.format(s)
        d = yaml.load(ys)
        yaml.dump(d, compare=ys)