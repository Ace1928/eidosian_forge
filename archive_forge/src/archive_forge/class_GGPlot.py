import packages is to use `importr()`, for example
import rpy2.robjects as robjects
import rpy2.robjects.constants
import rpy2.robjects.conversion as conversion
from rpy2.robjects.packages import importr, WeakPackage
from rpy2.robjects import rl
import warnings
class GGPlot(robjects.vectors.ListVector):
    """ A Grammar of Graphics Plot.

    GGPlot instances can be added to one an other in order to construct
    the final plot (the method `__add__()` is implemented).
    """
    _constructor = ggplot2._env['ggplot']
    _rprint = ggplot2._env['print.ggplot']
    _add = ggplot2._env['%+%']

    @classmethod
    def new(cls, data, mapping=_AES_RLANG, **kwargs):
        """ Constructor for the class GGplot. """
        data = conversion.get_conversion().py2rpy(data)
        res = cls(cls._constructor(data, mapping=mapping, **kwargs))
        return res

    def plot(self, vp=rpy2.robjects.constants.NULL):
        self._rprint(self, vp=vp)

    def __add__(self, obj):
        res = self._add(self, obj)
        if 'gg' not in res.rclass:
            raise ValueError("Added object did not give a ggplot result (get class '%s')." % res.rclass[0])
        return self.__class__(res)

    def save(self, filename, **kwargs):
        """ Save the plot ( calls R's `ggplot2::ggsave()` ) """
        ggplot2.ggsave(filename=filename, plot=self, **kwargs)