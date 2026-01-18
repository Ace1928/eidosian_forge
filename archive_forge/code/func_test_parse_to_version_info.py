import os
import subprocess
import sys
import pytest
import matplotlib
@pytest.mark.parametrize('version_str, version_tuple', [('3.5.0', (3, 5, 0, 'final', 0)), ('3.5.0rc2', (3, 5, 0, 'candidate', 2)), ('3.5.0.dev820+g6768ef8c4c', (3, 5, 0, 'alpha', 820)), ('3.5.0.post820+g6768ef8c4c', (3, 5, 1, 'alpha', 820))])
def test_parse_to_version_info(version_str, version_tuple):
    assert matplotlib._parse_to_version_info(version_str) == version_tuple