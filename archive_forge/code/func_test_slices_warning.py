import os
import numpy as np
import pytest
from ... import from_emcee
from ..helpers import _emcee_lnprior as emcee_lnprior
from ..helpers import _emcee_lnprob as emcee_lnprob
from ..helpers import (  # pylint: disable=unused-import
@pytest.mark.parametrize('slices', [[0, 0, slice(2, None)], [0, 1, slice(1, None)]])
def test_slices_warning(self, data, slices):
    with pytest.warns(UserWarning):
        from_emcee(data.obj, slices=slices)