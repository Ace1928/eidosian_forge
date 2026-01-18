from __future__ import print_function, unicode_literals
import sys
import pytest  # NOQA
import warnings  # NOQA
from pathlib import Path
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