import cudf
import pandas
import ray
from modin.core.execution.ray.common import RayWrapper
def apply_non_persistent(self, first, other, func, **kwargs):
    """
        Apply `func` to values associated with `first`/`other` keys of `self.cudf_dataframe_dict`.

        Parameters
        ----------
        first : int
            The first key associated with dataframe from `self.cudf_dataframe_dict`.
        other : int
            The second key associated with dataframe from `self.cudf_dataframe_dict`.
            If it isn't a real key, the `func` will be applied to the `first` only.
        func : callable
            A function to apply.
        **kwargs : dict
            Additional keywords arguments to be passed in `func`.

        Returns
        -------
        The type of return of `func`
            The result of the `func` (will be a ``ray.ObjectRef`` in outside level).
        """
    df1 = self.cudf_dataframe_dict[first]
    df2 = self.cudf_dataframe_dict[other] if other else None
    if not df2:
        result = func(df1, **kwargs)
    else:
        result = func(df1, df2, **kwargs)
    return result