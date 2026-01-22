from collections import defaultdict
from itertools import islice
import numpy as np
from scipy import sparse
from .base import TransformerMixin, _fit_context, clone
from .exceptions import NotFittedError
from .preprocessing import FunctionTransformer
from .utils import Bunch, _print_elapsed_time
from .utils._estimator_html_repr import _VisualBlock
from .utils._metadata_requests import METHODS
from .utils._param_validation import HasMethods, Hidden
from .utils._set_output import (
from .utils._tags import _safe_tags
from .utils.metadata_routing import (
from .utils.metaestimators import _BaseComposition, available_if
from .utils.parallel import Parallel, delayed
from .utils.validation import check_is_fitted, check_memory
def _parallel_func(self, X, y, fit_params, func):
    """Runs func in parallel on X and y"""
    self.transformer_list = list(self.transformer_list)
    self._validate_transformers()
    self._validate_transformer_weights()
    transformers = list(self._iter())
    params = Bunch(fit=fit_params, fit_transform=fit_params)
    return Parallel(n_jobs=self.n_jobs)((delayed(func)(transformer, X, y, weight, message_clsname='FeatureUnion', message=self._log_message(name, idx, len(transformers)), params=params) for idx, (name, transformer, weight) in enumerate(transformers, 1)))