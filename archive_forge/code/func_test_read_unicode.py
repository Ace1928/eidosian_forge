from __future__ import print_function
import sys
import textwrap
import pytest
from pathlib import Path
def test_read_unicode(self, tmpdir):
    from srsly.ruamel_yaml import YAML
    yaml = YAML()
    file_name = str(tmpdir) + '/tstFile.yaml'
    with open(file_name, 'wb') as fp:
        fp.write(u'text: HELLO_WORLD©\n'.encode('utf-8'))
    with open(file_name, 'r', encoding='utf8') as fp:
        text_dict = yaml.load(fp)
    assert text_dict['text'] == u'HELLO_WORLD©'