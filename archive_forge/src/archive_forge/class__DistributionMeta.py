import abc
import contextlib
import types
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import util
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
class _DistributionMeta(abc.ABCMeta):

    def __new__(mcs, classname, baseclasses, attrs):
        """Control the creation of subclasses of the Distribution class.

    The main purpose of this method is to properly propagate docstrings
    from private Distribution methods, like `_log_prob`, into their
    public wrappers as inherited by the Distribution base class
    (e.g. `log_prob`).

    Args:
      classname: The name of the subclass being created.
      baseclasses: A tuple of parent classes.
      attrs: A dict mapping new attributes to their values.

    Returns:
      The class object.

    Raises:
      TypeError: If `Distribution` is not a subclass of `BaseDistribution`, or
        the new class is derived via multiple inheritance and the first
        parent class is not a subclass of `BaseDistribution`.
      AttributeError:  If `Distribution` does not implement e.g. `log_prob`.
      ValueError:  If a `Distribution` public method lacks a docstring.
    """
        if not baseclasses:
            raise TypeError('Expected non-empty baseclass. Does Distribution not subclass _BaseDistribution?')
        which_base = [base for base in baseclasses if base == _BaseDistribution or issubclass(base, Distribution)]
        base = which_base[0]
        if base == _BaseDistribution:
            return abc.ABCMeta.__new__(mcs, classname, baseclasses, attrs)
        if not issubclass(base, Distribution):
            raise TypeError("First parent class declared for %s must be Distribution, but saw '%s'" % (classname, base.__name__))
        for attr in _DISTRIBUTION_PUBLIC_METHOD_WRAPPERS:
            special_attr = '_%s' % attr
            class_attr_value = attrs.get(attr, None)
            if attr in attrs:
                continue
            base_attr_value = getattr(base, attr, None)
            if not base_attr_value:
                raise AttributeError("Internal error: expected base class '%s' to implement method '%s'" % (base.__name__, attr))
            class_special_attr_value = attrs.get(special_attr, None)
            if class_special_attr_value is None:
                continue
            class_special_attr_docstring = tf_inspect.getdoc(class_special_attr_value)
            if not class_special_attr_docstring:
                continue
            class_attr_value = _copy_fn(base_attr_value)
            class_attr_docstring = tf_inspect.getdoc(base_attr_value)
            if class_attr_docstring is None:
                raise ValueError('Expected base class fn to contain a docstring: %s.%s' % (base.__name__, attr))
            class_attr_value.__doc__ = _update_docstring(class_attr_value.__doc__, 'Additional documentation from `%s`:\n\n%s' % (classname, class_special_attr_docstring))
            attrs[attr] = class_attr_value
        return abc.ABCMeta.__new__(mcs, classname, baseclasses, attrs)