from collections import abc as collections_abc
import logging
import sys
from typing import Mapping, Sequence, Text, TypeVar, Union
from .sequence import _is_attrs
from .sequence import _is_namedtuple
from .sequence import _sequence_like
from .sequence import _sorted
def _assert_shallow_structure(shallow_tree, input_tree, path=None, check_types=True):
    """Asserts that `shallow_tree` is a shallow structure of `input_tree`.

  That is, this function recursively tests if each key in shallow_tree has its
  corresponding key in input_tree.

  Examples:

  The following code will raise an exception:

  >>> shallow_tree = {"a": "A", "b": "B"}
  >>> input_tree = {"a": 1, "c": 2}
  >>> _assert_shallow_structure(shallow_tree, input_tree)
  Traceback (most recent call last):
    ...
  ValueError: The shallow_tree's keys are not a subset of the input_tree's ...

  The following code will raise an exception:

  >>> shallow_tree = ["a", "b"]
  >>> input_tree = ["c", ["d", "e"], "f"]
  >>> _assert_shallow_structure(shallow_tree, input_tree)
  Traceback (most recent call last):
    ...
  ValueError: The two structures don't have the same sequence length. ...

  By setting check_types=False, we drop the requirement that corresponding
  nodes in shallow_tree and input_tree have to be the same type. Sequences
  are treated equivalently to Mappables that map integer keys (indices) to
  values. The following code will therefore not raise an exception:

  >>> _assert_shallow_structure({0: "foo"}, ["foo"], check_types=False)

  Args:
    shallow_tree: an arbitrarily nested structure.
    input_tree: an arbitrarily nested structure.
    path: if not `None`, a tuple containing the current path in the nested
      structure. This is only used for more informative errror messages.
    check_types: if `True` (default) the sequence types of `shallow_tree` and
      `input_tree` have to be the same.

  Raises:
    TypeError: If `shallow_tree` is a sequence but `input_tree` is not.
    TypeError: If the sequence types of `shallow_tree` are different from
      `input_tree`. Only raised if `check_types` is `True`.
    ValueError: If the sequence lengths of `shallow_tree` are different from
      `input_tree`.
  """
    if is_nested(shallow_tree):
        if not is_nested(input_tree):
            if path is not None:
                raise TypeError(_IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ_WITH_PATH.format(path=list(path), input_type=type(input_tree)))
            else:
                raise TypeError(_IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ.format(type(input_tree)))
        if isinstance(shallow_tree, ObjectProxy):
            shallow_type = type(shallow_tree.__wrapped__)
        else:
            shallow_type = type(shallow_tree)
        if check_types and (not isinstance(input_tree, shallow_type)):
            shallow_is_namedtuple = _is_namedtuple(shallow_tree, False)
            input_is_namedtuple = _is_namedtuple(input_tree, False)
            if shallow_is_namedtuple and input_is_namedtuple:
                if not _tree.same_namedtuples(shallow_tree, input_tree):
                    raise TypeError(_STRUCTURES_HAVE_MISMATCHING_TYPES.format(input_type=type(input_tree), shallow_type=shallow_type))
            elif not (isinstance(shallow_tree, collections_abc.Mapping) and isinstance(input_tree, collections_abc.Mapping)):
                raise TypeError(_STRUCTURES_HAVE_MISMATCHING_TYPES.format(input_type=type(input_tree), shallow_type=shallow_type))
        if _num_elements(input_tree) != _num_elements(shallow_tree):
            raise ValueError(_STRUCTURES_HAVE_MISMATCHING_LENGTHS.format(input_length=_num_elements(input_tree), shallow_length=_num_elements(shallow_tree)))
        elif _num_elements(input_tree) < _num_elements(shallow_tree):
            raise ValueError(_INPUT_TREE_SMALLER_THAN_SHALLOW_TREE.format(input_size=_num_elements(input_tree), shallow_size=_num_elements(shallow_tree)))
        shallow_iter = _yield_sorted_items(shallow_tree)
        input_iter = _yield_sorted_items(input_tree)

        def get_matching_input_branch(shallow_key):
            for input_key, input_branch in input_iter:
                if input_key == shallow_key:
                    return input_branch
            raise ValueError(_SHALLOW_TREE_HAS_INVALID_KEYS.format([shallow_key]))
        for shallow_key, shallow_branch in shallow_iter:
            input_branch = get_matching_input_branch(shallow_key)
            _assert_shallow_structure(shallow_branch, input_branch, path + (shallow_key,) if path is not None else None, check_types=check_types)