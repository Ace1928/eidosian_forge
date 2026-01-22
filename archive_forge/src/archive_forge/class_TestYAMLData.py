from __future__ import print_function, unicode_literals
import sys
import pytest  # NOQA
import warnings  # NOQA
from pathlib import Path
class TestYAMLData(object):

    def yaml(self, yaml_version=None):
        from srsly.ruamel_yaml import YAML
        y = YAML()
        y.preserve_quotes = True
        if yaml_version:
            y.version = yaml_version
        return y

    def docs(self, path):
        from srsly.ruamel_yaml import YAML
        tyaml = YAML(typ='safe', pure=True)
        tyaml.register_class(YAMLData)
        tyaml.register_class(Python)
        tyaml.register_class(Output)
        tyaml.register_class(Assert)
        return list(tyaml.load_all(path))

    def yaml_load(self, value, yaml_version=None):
        yaml = self.yaml(yaml_version=yaml_version)
        data = yaml.load(value)
        return (yaml, data)

    def round_trip(self, input, output=None, yaml_version=None):
        from srsly.ruamel_yaml.compat import StringIO
        yaml, data = self.yaml_load(input.value, yaml_version=yaml_version)
        buf = StringIO()
        yaml.dump(data, buf)
        expected = input.value if output is None else output.value
        value = buf.getvalue()
        if PY2:
            value = value.decode('utf-8')
            print('value', value)
        assert value == expected

    def load_assert(self, input, confirm, yaml_version=None):
        from srsly.ruamel_yaml.compat import Mapping
        d = self.yaml_load(input.value, yaml_version=yaml_version)[1]
        print('confirm.value', confirm.value, type(confirm.value))
        if isinstance(confirm.value, Mapping):
            r = range(confirm.value['range'])
            lines = confirm.value['lines'].splitlines()
            for idx in r:
                for line in lines:
                    line = 'assert ' + line
                    print(line)
                    exec(line)
        else:
            for line in confirm.value.splitlines():
                line = 'assert ' + line
                print(line)
                exec(line)

    def run_python(self, python, data, tmpdir):
        from .roundtrip import save_and_run
        assert save_and_run(python.value, base_dir=tmpdir, output=data.value) == 0

    def test_yaml_data(self, yaml, tmpdir):
        from srsly.ruamel_yaml.compat import Mapping
        idx = 0
        typ = None
        yaml_version = None
        docs = self.docs(yaml)
        if isinstance(docs[0], Mapping):
            d = docs[0]
            typ = d.get('type')
            yaml_version = d.get('yaml_version')
            if 'python' in d:
                if not check_python_version(d['python']):
                    pytest.skip('unsupported version')
            idx += 1
        data = output = confirm = python = None
        for doc in docs[idx:]:
            if isinstance(doc, Output):
                output = doc
            elif isinstance(doc, Assert):
                confirm = doc
            elif isinstance(doc, Python):
                python = doc
                if typ is None:
                    typ = 'python_run'
            elif isinstance(doc, YAMLData):
                data = doc
            else:
                print('no handler for type:', type(doc), repr(doc))
                raise AssertionError()
        if typ is None:
            if data is not None and output is not None:
                typ = 'rt'
            elif data is not None and confirm is not None:
                typ = 'load_assert'
            else:
                assert data is not None
                typ = 'rt'
        print('type:', typ)
        if data is not None:
            print('data:', data.value, end='')
        print('output:', output.value if output is not None else output)
        if typ == 'rt':
            self.round_trip(data, output, yaml_version=yaml_version)
        elif typ == 'python_run':
            self.run_python(python, output if output is not None else data, tmpdir)
        elif typ == 'load_assert':
            self.load_assert(data, confirm, yaml_version=yaml_version)
        else:
            print('\nrun type unknown:', typ)
            raise AssertionError()