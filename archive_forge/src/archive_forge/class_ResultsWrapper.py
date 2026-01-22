import functools
import inspect
from textwrap import dedent
class ResultsWrapper:
    """
    Class which wraps a statsmodels estimation Results class and steps in to
    reattach metadata to results (if available)
    """
    _wrap_attrs = {}
    _wrap_methods = {}

    def __init__(self, results):
        self._results = results
        self.__doc__ = results.__doc__

    def __dir__(self):
        return [x for x in dir(self._results)]

    def __getattribute__(self, attr):
        get = lambda name: object.__getattribute__(self, name)
        try:
            results = get('_results')
        except AttributeError:
            pass
        try:
            return get(attr)
        except AttributeError:
            pass
        obj = getattr(results, attr)
        data = results.model.data
        how = self._wrap_attrs.get(attr)
        if how and isinstance(how, tuple):
            obj = data.wrap_output(obj, how[0], *how[1:])
        elif how:
            obj = data.wrap_output(obj, how=how)
        return obj

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict_):
        self.__dict__.update(dict_)

    def save(self, fname, remove_data=False):
        """
        Save a pickle of this instance.

        Parameters
        ----------
        fname : {str, handle}
            Either a filename or a valid file handle.
        remove_data : bool
            If False (default), then the instance is pickled without changes.
            If True, then all arrays with length nobs are set to None before
            pickling. See the remove_data method.
            In some cases not all arrays will be set to None.
        """
        from statsmodels.iolib.smpickle import save_pickle
        if remove_data:
            self.remove_data()
        save_pickle(self, fname)

    @classmethod
    def load(cls, fname):
        """
        Load a pickled results instance

        .. warning::

           Loading pickled models is not secure against erroneous or
           maliciously constructed data. Never unpickle data received from
           an untrusted or unauthenticated source.

        Parameters
        ----------
        fname : {str, handle}
            A string filename or a file handle.

        Returns
        -------
        Results
            The unpickled results instance.
        """
        from statsmodels.iolib.smpickle import load_pickle
        return load_pickle(fname)