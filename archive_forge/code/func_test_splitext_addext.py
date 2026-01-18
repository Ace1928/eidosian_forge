import pathlib
import pytest
from ..filename_parser import (
def test_splitext_addext():
    res = splitext_addext('fname.ext.gz')
    assert res == ('fname', '.ext', '.gz')
    res = splitext_addext('fname.ext')
    assert res == ('fname', '.ext', '')
    res = splitext_addext('fname.ext.foo', ('.foo', '.bar'))
    assert res == ('fname', '.ext', '.foo')
    res = splitext_addext('fname.ext.FOO', ('.foo', '.bar'))
    assert res == ('fname', '.ext', '.FOO')
    res = splitext_addext('fname.ext.FOO', ('.foo', '.bar'), True)
    assert res == ('fname.ext', '.FOO', '')
    res = splitext_addext('.nii')
    assert res == ('', '.nii', '')
    res = splitext_addext('...nii')
    assert res == ('..', '.nii', '')
    res = splitext_addext('.')
    assert res == ('.', '', '')
    res = splitext_addext('..')
    assert res == ('..', '', '')
    res = splitext_addext('...')
    assert res == ('...', '', '')