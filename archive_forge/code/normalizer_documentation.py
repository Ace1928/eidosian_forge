from typing import List
import numpy as np
import pandas as pd
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
Scales each sample to have unit norm.

    This preprocessor works by dividing each sample (i.e., row) by the sample's norm.
    The general formula is given by

    .. math::

        s' = \frac{s}{\lVert s \rVert_p}

    where :math:`s` is the sample, :math:`s'` is the transformed sample,
    :math:\lVert s \rVert`, and :math:`p` is the norm type.

    The following norms are supported:

    * `"l1"` (:math:`L^1`): Sum of the absolute values.
    * `"l2"` (:math:`L^2`): Square root of the sum of the squared values.
    * `"max"` (:math:`L^\infty`): Maximum value.

    Examples:
        >>> import pandas as pd
        >>> import ray
        >>> from ray.data.preprocessors import Normalizer
        >>>
        >>> df = pd.DataFrame({"X1": [1, 1], "X2": [1, 0], "X3": [0, 1]})
        >>> ds = ray.data.from_pandas(df)  # doctest: +SKIP
        >>> ds.to_pandas()  # doctest: +SKIP
           X1  X2  X3
        0   1   1   0
        1   1   0   1

        The :math:`L^2`-norm of the first sample is :math:`\sqrt{2}`, and the
        :math:`L^2`-norm of the second sample is :math:`1`.

        >>> preprocessor = Normalizer(columns=["X1", "X2"])
        >>> preprocessor.fit_transform(ds).to_pandas()  # doctest: +SKIP
                 X1        X2  X3
        0  0.707107  0.707107   0
        1  1.000000  0.000000   1

        The :math:`L^1`-norm of the first sample is :math:`2`, and the
        :math:`L^1`-norm of the second sample is :math:`1`.

        >>> preprocessor = Normalizer(columns=["X1", "X2"], norm="l1")
        >>> preprocessor.fit_transform(ds).to_pandas()  # doctest: +SKIP
            X1   X2  X3
        0  0.5  0.5   0
        1  1.0  0.0   1

        The :math:`L^\infty`-norm of the both samples is :math:`1`.

        >>> preprocessor = Normalizer(columns=["X1", "X2"], norm="max")
        >>> preprocessor.fit_transform(ds).to_pandas()  # doctest: +SKIP
            X1   X2  X3
        0  1.0  1.0   0
        1  1.0  0.0   1

    Args:
        columns: The columns to scale. For each row, these colmumns are scaled to
            unit-norm.
        norm: The norm to use. The supported values are ``"l1"``, ``"l2"``, or
            ``"max"``. Defaults to ``"l2"``.

    Raises:
        ValueError: if ``norm`` is not ``"l1"``, ``"l2"``, or ``"max"``.
    