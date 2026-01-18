from typing import Any, Callable, Optional, overload, Union
from typing_extensions import Protocol
from cirq import protocols, _compat
def value_equality(cls: Optional[type]=None, *, unhashable: bool=False, distinct_child_types: bool=False, manual_cls: bool=False, approximate: bool=False) -> Union[Callable[[type], type], type]:
    """Implements __eq__/__ne__/__hash__ via a _value_equality_values_ method.

    _value_equality_values_ is a method that the decorated class must implement.

    _value_equality_approximate_values_ is a method that the decorated class
    might implement if special support for approximate equality is required.
    This is only used when approximate argument is set. When approximate
    argument is set and _value_equality_approximate_values_ is not defined,
    _value_equality_values_ values are used for approximate equality.
    For example, this can be used to compare periodic values like angles: the
    angle value can be wrapped with `PeriodicValue`. When returned as part of
    approximate values a special normalization will be done automatically to
    guarantee correctness.

    Note that the type of the decorated value is included as part of the value
    equality values. This is so that completely separate classes with identical
    equality values (e.g. a Point2D and a Vector2D) don't compare as equal.
    Further note that this means that child types of the decorated type will be
    considered equal to each other, though this behavior can be changed via
    the 'distinct_child_types` argument. The type logic is implemented behind
    the scenes by a `_value_equality_values_cls_` method added to the class.

    Args:
        cls: The type to decorate. Automatically passed in by python when using
            the @cirq.value_equality decorator notation on a class.
        unhashable: When set, the __hash__ method will be set to None instead of
            to a hash of the equality class and equality values. Useful for
            mutable types such as dictionaries.
        distinct_child_types: When set, classes that inherit from the decorated
            class will not be considered equal to it. Also, different child
            classes will not be considered equal to each other. Useful for when
            the decorated class is an abstract class or trait that is helping to
            define equality for many conceptually distinct concrete classes.
        manual_cls: When set, the method '_value_equality_values_cls_' must be
            implemented. This allows a new class to compare as equal to another
            existing class that is also using value equality, by having the new
            class return the existing class' type.
            Incompatible with `distinct_child_types`.
        approximate: When set, the decorated class will be enhanced with
            `_approx_eq_` implementation and thus start to support the
            `SupportsApproximateEquality` protocol.

    Raises:
        TypeError: If the class decorated does not implement the required
            `_value_equality_values` method or, if `manual_cls` is True,
            the class does not implement `_value_equality_values_cls_`.
        ValueError: If both `distinct_child_types` and `manual_cls` are
            specified.
    """
    if cls is None:
        return lambda deferred_cls: value_equality(deferred_cls, unhashable=unhashable, manual_cls=manual_cls, distinct_child_types=distinct_child_types, approximate=approximate)
    if distinct_child_types and manual_cls:
        raise ValueError("'distinct_child_types' is incompatible with 'manual_cls")
    values_getter = getattr(cls, '_value_equality_values_', None)
    if values_getter is None:
        raise TypeError('The @cirq.value_equality decorator requires a _value_equality_values_ method to be defined.')
    if distinct_child_types:
        setattr(cls, '_value_equality_values_cls_', lambda self: type(self))
    elif manual_cls:
        cls_getter = getattr(cls, '_value_equality_values_cls_', None)
        if cls_getter is None:
            raise TypeError('The @cirq.value_equality decorator requires a _value_equality_values_cls_ method to be defined when "manual_cls" is set.')
    else:
        setattr(cls, '_value_equality_values_cls_', lambda self: cls)
    cached_values_getter = values_getter if unhashable else _compat.cached_method(values_getter)
    setattr(cls, '_value_equality_values_', cached_values_getter)
    setattr(cls, '__hash__', None if unhashable else _compat.cached_method(_value_equality_hash))
    setattr(cls, '__eq__', _value_equality_eq)
    setattr(cls, '__ne__', _value_equality_ne)
    if approximate:
        if not hasattr(cls, '_value_equality_approximate_values_'):
            setattr(cls, '_value_equality_approximate_values_', cached_values_getter)
        else:
            approx_values_getter = getattr(cls, '_value_equality_approximate_values_')
            cached_approx_values_getter = approx_values_getter if unhashable else _compat.cached_method(approx_values_getter)
            setattr(cls, '_value_equality_approximate_values_', cached_approx_values_getter)
        setattr(cls, '_approx_eq_', _value_equality_approx_eq)
    return cls