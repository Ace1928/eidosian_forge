import sys
from .interactive import Interactive
class DaskInteractive(Interactive):

    @classmethod
    def applies(cls, obj):
        if 'dask.dataframe' in sys.modules:
            import dask.dataframe as dd
            return isinstance(obj, (dd.Series, dd.DataFrame))
        return False

    def compute(self):
        self._method = 'compute'
        return self.__call__()