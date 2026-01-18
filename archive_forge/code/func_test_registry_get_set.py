import pytest
import sys
from pathlib import Path
import catalogue
def test_registry_get_set():
    test_registry = catalogue.create('test')
    with pytest.raises(catalogue.RegistryError):
        test_registry.get('foo')
    test_registry.register('foo', func=lambda x: x)
    assert 'foo' in test_registry