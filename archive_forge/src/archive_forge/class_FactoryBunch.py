from statsmodels.compat.pandas import testing as pdt
import numpy.testing as npt
import pandas
from statsmodels.tools.tools import Bunch
class FactoryBunch(Bunch):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, attribute):
            raise AttributeError('{} is required and must be passed to the constructor'.format(attribute))
        for i, att in enumerate(columns):
            self[att] = getattr(self, attribute)[:, i]