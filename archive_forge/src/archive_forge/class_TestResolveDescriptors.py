from __future__ import annotations
import sys
import types
from typing import Any
import pytest
import numpy as np
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
class TestResolveDescriptors:
    method = get_castingimpl(type(np.dtype('d')), type(np.dtype('f')))

    @pytest.mark.parametrize('args', [(True,), (None,), ((None, None, None),), ((None, None),), ((np.dtype('d'), True),), ((np.dtype('f'), None),)])
    def test_invalid_arguments(self, args):
        with pytest.raises(TypeError):
            self.method._resolve_descriptors(*args)