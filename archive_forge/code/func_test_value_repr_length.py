import io
import os
from ipywidgets import Image
import hashlib
import pkgutil
import tempfile
from contextlib import contextmanager
def test_value_repr_length():
    with get_logo_png() as LOGO_PNG:
        with open(LOGO_PNG, 'rb') as f:
            img = Image.from_file(f)
            assert len(img.__repr__()) < 140
            assert img.__repr__().endswith(')')
            assert img.__repr__()[-5:-2] == '...'