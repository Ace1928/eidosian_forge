import copy
import functools
import inspect
import platform
import re
import warnings
from collections import defaultdict
import numpy as np
from . import __version__
from ._config import config_context, get_config
from .exceptions import InconsistentVersionWarning
from .utils import _IS_32BIT
from .utils._estimator_html_repr import _HTMLDocumentationLinkMixin, estimator_html_repr
from .utils._metadata_requests import _MetadataRequester, _routing_enabled
from .utils._param_validation import validate_parameter_constraints
from .utils._set_output import _SetOutputMixin
from .utils._tags import (
from .utils.validation import (
class _UnstableArchMixin:
    """Mark estimators that are non-determinstic on 32bit or PowerPC"""

    def _more_tags(self):
        return {'non_deterministic': _IS_32BIT or platform.machine().startswith(('ppc', 'powerpc'))}