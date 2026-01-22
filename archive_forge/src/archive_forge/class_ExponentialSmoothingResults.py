import numpy as np
import pandas as pd
from statsmodels.base.data import PandasData
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.validation import (array_like, bool_like, float_like,
from statsmodels.tsa.exponential_smoothing import initialization as es_init
from statsmodels.tsa.statespace import initialization as ss_init
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.compat.pandas import Appender
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import forg
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
class ExponentialSmoothingResults(MLEResults):
    """
    Results from fitting a linear exponential smoothing model
    """

    def __init__(self, model, params, filter_results, cov_type=None, **kwargs):
        super().__init__(model, params, filter_results, cov_type, **kwargs)
        self.initial_state = model._initial_state
        if isinstance(self.data, PandasData):
            index = self.data.row_labels
            self.initial_state = pd.DataFrame([model._initial_state], columns=model.state_names[1:])
            if model._index_dates and model._index_freq is not None:
                self.initial_state.index = index.shift(-1)[:1]

    @Appender(MLEResults.summary.__doc__)
    def summary(self, alpha=0.05, start=None):
        specification = ['A']
        if self.model.trend and self.model.damped_trend:
            specification.append('Ad')
        elif self.model.trend:
            specification.append('A')
        else:
            specification.append('N')
        if self.model.seasonal:
            specification.append('A')
        else:
            specification.append('N')
        model_name = 'ETS(' + ', '.join(specification) + ')'
        summary = super().summary(alpha=alpha, start=start, title='Exponential Smoothing Results', model_name=model_name)
        if self.model.initialization_method != 'estimated':
            params = np.array(self.initial_state)
            if params.ndim > 1:
                params = params[0]
            names = self.model.state_names[1:]
            param_header = ['initialization method: %s' % self.model.initialization_method]
            params_stubs = names
            params_data = [[forg(params[i], prec=4)] for i in range(len(params))]
            initial_state_table = SimpleTable(params_data, param_header, params_stubs, txt_fmt=fmt_params)
            summary.tables.insert(-1, initial_state_table)
        return summary