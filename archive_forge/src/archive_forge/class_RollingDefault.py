from .default import DefaultMethod
class RollingDefault(DefaultMethod):
    """Builder for default-to-pandas aggregation on a rolling window functions."""
    OBJECT_TYPE = 'Rolling'

    @classmethod
    def _build_rolling(cls, func):
        """
        Build function that creates a rolling window and executes `func` on it.

        Parameters
        ----------
        func : callable
            Function to execute on a rolling window.

        Returns
        -------
        callable
            Function that takes pandas DataFrame and applies `func` on a rolling window.
        """

        def fn(df, rolling_kwargs, *args, **kwargs):
            """Create rolling window for the passed frame and execute specified `func` on it."""
            roller = df.rolling(**rolling_kwargs)
            if type(func) is property:
                return func.fget(roller)
            return func(roller, *args, **kwargs)
        return fn

    @classmethod
    def register(cls, func, **kwargs):
        """
        Build function that do fallback to pandas to apply `func` on a rolling window.

        Parameters
        ----------
        func : callable
            Function to execute on a rolling window.
        **kwargs : kwargs
            Additional arguments that will be passed to function builder.

        Returns
        -------
        callable
            Function that takes query compiler and defaults to pandas to apply aggregation
            `func` on a rolling window.
        """
        return super().register(cls._build_rolling(func), fn_name=func.__name__, **kwargs)