import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_pre_and_postprocess_hooks(self):
    pre_backup = operation._preprocess_hooks
    post_backup = operation._postprocess_hooks
    operation._preprocess_hooks = [lambda op, x: {'label': str(x.id)}]
    operation._postprocess_hooks = [lambda op, x, **kwargs: x.clone(**kwargs)]
    curve = Curve([1, 2, 3])
    self.assertEqual(operation(curve).label, str(curve.id))
    operation._preprocess_hooks = pre_backup
    operation._postprocess_hooks = post_backup