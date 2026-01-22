import numpy as np
import pandas as pd
import param
from packaging.version import Version
from ..core import Element, Operation
from ..core.data import PandasInterface
from ..core.util import _PANDAS_FUNC_LOOKUP, pandas_version
from ..element import Scatter
class resample(Operation):
    """
    Resamples a timeseries of dates with a frequency and function.
    """
    closed = param.ObjectSelector(default=None, objects=['left', 'right'], doc='Which side of bin interval is closed', allow_None=True)
    function = param.Callable(default=np.mean, doc='\n        Function for computing new values out of existing ones.')
    label = param.ObjectSelector(default='right', doc='\n        The bin edge to label the bin with.')
    rule = param.String(default='D', doc='\n        A string representing the time interval over which to apply the resampling')

    def _process_layer(self, element, key=None):
        df = PandasInterface.as_dframe(element)
        xdim = element.kdims[0].name
        resample_kwargs = {'rule': self.p.rule, 'label': self.p.label, 'closed': self.p.closed}
        df = df.set_index(xdim).resample(**resample_kwargs)
        fn = _PANDAS_FUNC_LOOKUP.get(self.p.function, self.p.function)
        return element.clone(df.apply(fn).reset_index())

    def _process(self, element, key=None):
        return element.map(self._process_layer, Element)