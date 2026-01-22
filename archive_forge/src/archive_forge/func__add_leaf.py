from collections.abc import Iterable
from io import StringIO
from numbers import Integral
import numpy as np
from ..base import is_classifier
from ..utils._param_validation import HasMethods, Interval, StrOptions, validate_params
from ..utils.validation import check_array, check_is_fitted
from . import DecisionTreeClassifier, DecisionTreeRegressor, _criterion, _tree
from ._reingold_tilford import Tree, buchheim
def _add_leaf(value, weighted_n_node_samples, class_name, indent):
    val = ''
    if isinstance(decision_tree, DecisionTreeClassifier):
        if show_weights:
            val = ['{1:.{0}f}, '.format(decimals, v * weighted_n_node_samples) for v in value]
            val = '[' + ''.join(val)[:-2] + ']'
            weighted_n_node_samples
        val += ' class: ' + str(class_name)
    else:
        val = ['{1:.{0}f}, '.format(decimals, v) for v in value]
        val = '[' + ''.join(val)[:-2] + ']'
    export_text.report += value_fmt.format(indent, '', val)