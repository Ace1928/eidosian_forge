from __future__ import print_function
import sys
import textwrap
import pytest
from pathlib import Path
def test_write_unicode(self, tmpdir):
    from srsly.ruamel_yaml import YAML
    yaml = YAML()
    text_dict = {'text': u'HELLO_WORLD©'}
    file_name = str(tmpdir) + '/tstFile.yaml'
    yaml.dump(text_dict, open(file_name, 'w', encoding='utf8', newline='\n'))
    assert open(file_name, 'rb').read().decode('utf-8') == u'text: HELLO_WORLD©\n'