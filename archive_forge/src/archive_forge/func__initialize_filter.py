import contextlib
from warnings import warn
import numpy as np
from .representation import OptionWrapper, Representation, FrozenRepresentation
from .tools import reorder_missing_matrix, reorder_missing_vector
from . import tools
from statsmodels.tools.sm_exceptions import ValueWarning
def _initialize_filter(self, filter_method=None, inversion_method=None, stability_method=None, conserve_memory=None, tolerance=None, filter_timing=None, loglikelihood_burn=None):
    if filter_method is None:
        filter_method = self.filter_method
    if inversion_method is None:
        inversion_method = self.inversion_method
    if stability_method is None:
        stability_method = self.stability_method
    if conserve_memory is None:
        conserve_memory = self.conserve_memory
    if loglikelihood_burn is None:
        loglikelihood_burn = self.loglikelihood_burn
    if filter_timing is None:
        filter_timing = self.filter_timing
    if tolerance is None:
        tolerance = self.tolerance
    if self.endog is None:
        raise RuntimeError('Must bind a dataset to the model before filtering or smoothing.')
    prefix, dtype, create_statespace = self._initialize_representation()
    create_filter = create_statespace or prefix not in self._kalman_filters
    if not create_filter:
        kalman_filter = self._kalman_filters[prefix]
        create_filter = not kalman_filter.conserve_memory == conserve_memory or not kalman_filter.loglikelihood_burn == loglikelihood_burn
    if create_filter:
        if prefix in self._kalman_filters:
            del self._kalman_filters[prefix]
        cls = self.prefix_kalman_filter_map[prefix]
        self._kalman_filters[prefix] = cls(self._statespaces[prefix], filter_method, inversion_method, stability_method, conserve_memory, filter_timing, tolerance, loglikelihood_burn)
    else:
        kalman_filter = self._kalman_filters[prefix]
        kalman_filter.set_filter_method(filter_method, False)
        kalman_filter.inversion_method = inversion_method
        kalman_filter.stability_method = stability_method
        kalman_filter.filter_timing = filter_timing
        kalman_filter.tolerance = tolerance
    return (prefix, dtype, create_filter, create_statespace)