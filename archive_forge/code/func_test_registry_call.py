import pytest
import sys
from pathlib import Path
import catalogue
def test_registry_call():
    test_registry = catalogue.create('test')
    test_registry('foo', func=lambda x: x)
    assert 'foo' in test_registry