import collections
import threading
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util.tf_export import tf_export
Converts `value` to a `Tensor` using registered conversion functions.

  Args:
    value: An object whose type has a registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor. If missing, the type
      is inferred from the type of `value`.
    name: Optional name to use if a new `Tensor` is created.
    as_ref: Optional boolean specifying if the returned value should be a
      reference-type `Tensor` (e.g. Variable). Pass-through to the registered
      conversion function. Defaults to `False`.
    preferred_dtype: Optional element type for the returned tensor.
      Used when dtype is None. In some cases, a caller may not have a dtype
      in mind when converting to a tensor, so `preferred_dtype` can be used
      as a soft preference. If the conversion to `preferred_dtype` is not
      possible, this argument has no effect.
    accepted_result_types: Optional collection of types as an allow-list
      for the returned value. If a conversion function returns an object
      which is not an instance of some type in this collection, that value
      will not be returned.

  Returns:
    A `Tensor` converted from `value`.

  Raises:
    ValueError: If `value` is a `Tensor` and conversion is requested
      to a `Tensor` with an incompatible `dtype`.
    TypeError: If no conversion function is registered for an element in
      `values`.
    RuntimeError: If a registered conversion function returns an invalid
      value.
  