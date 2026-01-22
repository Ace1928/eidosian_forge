import warnings
import numpy as np
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils._param_validation import StrOptions
from ..utils._set_output import ADAPTERS_MANAGER, _get_output_config
from ..utils.metaestimators import available_if
from ..utils.validation import (
def _get_adapter_from_container(container):
    """Get the adapter that nows how to handle such container.

    See :class:`sklearn.utils._set_output.ContainerAdapterProtocol` for more
    details.
    """
    module_name = container.__class__.__module__.split('.')[0]
    try:
        return ADAPTERS_MANAGER.adapters[module_name]
    except KeyError as exc:
        available_adapters = list(ADAPTERS_MANAGER.adapters.keys())
        raise ValueError(f'The container does not have a registered adapter in scikit-learn. Available adapters are: {available_adapters} while the container provided is: {container!r}.') from exc