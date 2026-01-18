import cudf
import pandas
import ray
from modin.core.execution.ray.common import RayWrapper

        Convert `pandas_df` to ``cudf.DataFrame`` and put it to `self.cudf_dataframe_dict`.

        Parameters
        ----------
        pandas_df : pandas.DataFrame/pandas.Series
            A pandas DataFrame/Series to be added.

        Returns
        -------
        int
            The key associated with added dataframe
            (will be a ``ray.ObjectRef`` in outside level).
        