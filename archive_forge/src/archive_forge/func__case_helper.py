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
def _case_helper(cond_fn, pred_fn_pairs, default, exclusive, name, allow_python_preds=False, **cond_kwargs):
    """Implementation of case that allows for different cond functions.

  Args:
    cond_fn: method that has signature and semantics of `cond` above.
    pred_fn_pairs: Dict or list of pairs of a boolean scalar tensor, and a
      callable which returns a list of tensors.
    default: Optional callable that returns a list of tensors.
    exclusive: True iff at most one predicate is allowed to evaluate to `True`.
    name: A name for this operation (optional).
    allow_python_preds: if true, pred_fn_pairs may contain Python bools in
      addition to boolean Tensors
    **cond_kwargs: keyword arguments that will be passed to `cond_fn`.

  Returns:
    The tensors returned by the first pair whose predicate evaluated to True, or
    those returned by `default` if none does.

  Raises:
    TypeError: If `pred_fn_pairs` is not a list/dictionary.
    TypeError: If `pred_fn_pairs` is a list but does not contain 2-tuples.
    TypeError: If `fns[i]` is not callable for any i, or `default` is not
               callable.
  """
    predicates, actions = _case_verify_and_canonicalize_args(pred_fn_pairs, exclusive, name, allow_python_preds)
    with ops.name_scope(name, 'case', [predicates]):
        if default is None:
            default, predicates, actions = _case_create_default_action(predicates, actions)
        fn = default
        for predicate, action in reversed(list(zip(predicates, actions))):
            fn = functools.partial(cond_fn, predicate, true_fn=action, false_fn=fn, **cond_kwargs)
        if exclusive:
            with ops.control_dependencies([_assert_at_most_n_true(predicates, n=1, msg='Input error: exclusive=True')]):
                return fn()
        else:
            return fn()