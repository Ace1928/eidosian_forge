import collections
import functools
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _case_verify_and_canonicalize_args(pred_fn_pairs, exclusive, name, allow_python_preds):
    """Verifies input arguments for the case function.

  Args:
    pred_fn_pairs: Dict or list of pairs of a boolean scalar tensor, and a
      callable which returns a list of tensors.
    exclusive: True iff at most one predicate is allowed to evaluate to `True`.
    name: A name for the case operation.
    allow_python_preds: if true, pred_fn_pairs may contain Python bools in
      addition to boolean Tensors

  Raises:
    TypeError: If `pred_fn_pairs` is not a list/dictionary.
    TypeError: If `pred_fn_pairs` is a list but does not contain 2-tuples.
    TypeError: If `fns[i]` is not callable for any i, or `default` is not
               callable.

  Returns:
    a tuple <list of scalar bool tensors, list of callables>.
  """
    if not isinstance(pred_fn_pairs, (list, tuple, dict)):
        raise TypeError(f"'pred_fn_pairs' must be a list, tuple, or dict. Received: {type(pred_fn_pairs)}")
    if isinstance(pred_fn_pairs, collections.OrderedDict):
        pred_fn_pairs = pred_fn_pairs.items()
    elif isinstance(pred_fn_pairs, dict):
        if context.executing_eagerly():
            if not exclusive:
                raise ValueError("Unordered dictionaries are not supported for the 'pred_fn_pairs' argument when `exclusive=False` and eager mode is enabled.")
            pred_fn_pairs = list(pred_fn_pairs.items())
        else:
            pred_fn_pairs = sorted(pred_fn_pairs.items(), key=lambda item: item[0].name)
            if not exclusive:
                logging.warn('%s: An unordered dictionary of predicate/fn pairs was provided, but exclusive=False. The order of conditional tests is deterministic but not guaranteed.', name)
    for pred_fn_pair in pred_fn_pairs:
        if not isinstance(pred_fn_pair, tuple) or len(pred_fn_pair) != 2:
            raise TypeError(f"Each entry in 'pred_fn_pairs' must be a 2-tuple. Received {pred_fn_pair}.")
        pred, fn = pred_fn_pair
        if isinstance(pred, tensor.Tensor):
            if pred.dtype != dtypes.bool:
                raise TypeError('pred must be Tensor of type bool: %s' % pred.name)
        elif not allow_python_preds:
            raise TypeError('pred must be a Tensor, got: %s' % pred)
        elif not isinstance(pred, bool):
            raise TypeError('pred must be a Tensor or bool, got: %s' % pred)
        if not callable(fn):
            raise TypeError('fn for pred %s must be callable.' % pred.name)
    predicates, actions = zip(*pred_fn_pairs)
    return (predicates, actions)