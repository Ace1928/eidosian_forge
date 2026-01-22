import logging
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
Combine numeric columns into a column of type
    :class:`~ray.air.util.tensor_extensions.pandas.TensorDtype`.

    This preprocessor concatenates numeric columns and stores the result in a new
    column. The new column contains
    :class:`~ray.air.util.tensor_extensions.pandas.TensorArrayElement` objects of
    shape :math:`(m,)`, where :math:`m` is the number of columns concatenated.
    The :math:`m` concatenated columns are dropped after concatenation.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> import ray
        >>> from ray.data.preprocessors import Concatenator

        :py:class:`Concatenator` combines numeric columns into a column of
        :py:class:`~ray.air.util.tensor_extensions.pandas.TensorDtype`.

        >>> df = pd.DataFrame({"X0": [0, 3, 1], "X1": [0.5, 0.2, 0.9]})
        >>> ds = ray.data.from_pandas(df)  # doctest: +SKIP
        >>> concatenator = Concatenator()
        >>> concatenator.fit_transform(ds).to_pandas()  # doctest: +SKIP
           concat_out
        0  [0.0, 0.5]
        1  [3.0, 0.2]
        2  [1.0, 0.9]

        By default, the created column is called `"concat_out"`, but you can specify
        a different name.

        >>> concatenator = Concatenator(output_column_name="tensor")
        >>> concatenator.fit_transform(ds).to_pandas()  # doctest: +SKIP
               tensor
        0  [0.0, 0.5]
        1  [3.0, 0.2]
        2  [1.0, 0.9]

        Sometimes, you might not want to concatenate all of of the columns in your
        dataset. In this case, you can exclude columns with the ``exclude`` parameter.

        >>> df = pd.DataFrame({"X0": [0, 3, 1], "X1": [0.5, 0.2, 0.9], "Y": ["blue", "orange", "blue"]})
        >>> ds = ray.data.from_pandas(df)  # doctest: +SKIP
        >>> concatenator = Concatenator(exclude=["Y"])
        >>> concatenator.fit_transform(ds).to_pandas()  # doctest: +SKIP
                Y  concat_out
        0    blue  [0.0, 0.5]
        1  orange  [3.0, 0.2]
        2    blue  [1.0, 0.9]

        Alternatively, you can specify which columns to concatenate with the
        ``include`` parameter.

        >>> concatenator = Concatenator(include=["X0", "X1"])
        >>> concatenator.fit_transform(ds).to_pandas()  # doctest: +SKIP
                Y  concat_out
        0    blue  [0.0, 0.5]
        1  orange  [3.0, 0.2]
        2    blue  [1.0, 0.9]

        Note that if a column is in both ``include`` and ``exclude``, the column is
        excluded.

        >>> concatenator = Concatenator(include=["X0", "X1", "Y"], exclude=["Y"])
        >>> concatenator.fit_transform(ds).to_pandas()  # doctest: +SKIP
                Y  concat_out
        0    blue  [0.0, 0.5]
        1  orange  [3.0, 0.2]
        2    blue  [1.0, 0.9]

        By default, the concatenated tensor is a ``dtype`` common to the input columns.
        However, you can also explicitly set the ``dtype`` with the ``dtype``
        parameter.

        >>> concatenator = Concatenator(include=["X0", "X1"], dtype=np.float32)
        >>> concatenator.fit_transform(ds)  # doctest: +SKIP
        Dataset(num_blocks=1, num_rows=3, schema={Y: object, concat_out: TensorDtype(shape=(2,), dtype=float32)})

    Args:
        output_column_name: The desired name for the new column.
            Defaults to ``"concat_out"``.
        include: A list of columns to concatenate. If ``None``, all columns are
            concatenated.
        exclude: A list of column to exclude from concatenation.
            If a column is in both ``include`` and ``exclude``, the column is excluded
            from concatenation.
        dtype: The ``dtype`` to convert the output tensors to. If unspecified,
            the ``dtype`` is determined by standard coercion rules.
        raise_if_missing: If ``True``, an error is raised if any
            of the columns in ``include`` or ``exclude`` don't exist.
            Defaults to ``False``.

    Raises:
        ValueError: if `raise_if_missing` is `True` and a column in `include` or
            `exclude` doesn't exist in the dataset.
    