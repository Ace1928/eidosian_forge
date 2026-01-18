# Licensed under the GPL: https://www.gnu.org/licenses/gpl-3.0.en.html
# For details: reprotest/debian/copyright
"""Merge-diff config.

This module implements a basic generic language for representing an object
tree, described via a sequence of addition and subtraction operations on
different parts of the tree, that are merged together to form the new tree.

This is useful for manually specifying object trees in a compact way. This
saves human users time instead of typing the same thing many many times.

This module is totally independent of reprotest.
"""

import collections
import functools
import re
import sys
import types


def rgetattr2(obj, attr, one=None):
    if one is not None:
        def _getattr2(arg, attr):
            obj, one = arg
            val = getattr(one, attr)
            return (getattr(obj, attr) if hasattr(obj, attr) else val), val
    else:
        def _getattr2(arg, attr):
            obj, one = arg
            val = getattr(obj, attr)
            return val, val
    return functools.reduce(_getattr2, attr.split('.') if attr else [], (obj, one))

def rsetattr(obj, attr, val, one=None):
    pre, _, post = attr.rpartition('.')
    target, target_one = rgetattr2(obj, pre, one) if pre else (obj, one)
    target = target._replace(**{post: val})
    return rsetattr(obj, pre, target) if pre else target

def rdelattr(obj, attr, zero=None):
    pre, _, post = attr.rpartition('.')
    target, target_zero = rgetattr2(obj, pre, zero) if pre else (obj, zero)
    try:
        target = target._delete(post)
    except AttributeError:
        if zero is not None:
            target = target._replace(**{post: rgetattr2(target_zero, post)[0]})
        else:
            raise
    return rsetattr(obj, pre, target) if pre else target

class ImmutableNamespace(types.SimpleNamespace):
    def _replace(self, **kwargs):
        new = self.__dict__.copy()
        new.update(**kwargs)
        return self.__class__(**new)

    def _delete(self, name):
        new = self.__dict__.copy()
        if name in new:
            del new[name]
        return self.__class__(**new)

def strlist_set(sep, value=[]):
    class strlist_set(list):
        def __init__(self, value=[]):
            if isinstance(value, str):
                value = value.split(sep) if value else []
            return super().__init__(value)

        def dedup(self, seq):
            seen = set()
            seen_add = seen.add
            return self.__class__([x for x in seq if not (x in seen or seen_add(x))])

        def __add__(self, value):
            return self.dedup(super().__add__(value))

        def __iadd__(self, value):
            return self.__add__(value)

        def __sub__(self, value):
            return self.dedup([x for x in self if x not in set(value)])

        def __isub__(self, value):
            return self.__sub__(value)

        def sep(self):
            return sep
    return strlist_set(value)

def parse(d, action, one, zero=None, aliases={}):
    """Parse an action, apply it to an object and return the new value.

    Args:

    obj: The top-level object to apply the action to.

    action: The action to apply, specified as a string. The string is split on
        the leftmost occurrence of any supported $operator; everything to the
        left of it is the $attribute and to the right of it is the $operand.

        $attribute is in dotted-notation and specifies which $target attribute
        (or sub-attribute, etc) of the $obj to apply $operator and $operand to.
        If $attribute is the empty string, we apply these to $obj itself. If
        the actual attribute does not exist on $obj, we implicitly add it as if
        it had just been added with the "+" operator.

        Supported operators:
        +
            Add the attribute named $operand to the $target. The value of the
            attribute is taken from the $one default object. However if the
            attribute is already on $target, it retains its existing value.
        -
            Remove the attribute named $operand from the $target. If it does
            not support removing attributes, instead we set its value to that
            taken from the $zero default object.
        @
            Re-set the attribute to its default value from $one. This is just a
            shortcut for doing - and then +.
        =, +=, -=
            Apply the $operand to $target using the normal Python =, += and -=
            operators respectively. If $operand is not of the same type as
            $target, it is first coerced into it, using its class constructor.
        ++, --
            Apply 1 to the $target using the operators +=, -= respectively.

    one: Having the same structure as $obj, this is used for default values
        when adding new attributes; see "+" operator.

    zero: Having the same structure as $obj. this is used for default values
        when it is not possible to remove an attribute; see "-" operator.

    aliases: A dictionary specifying aliases for various operations. Currently
        supports the following keys:

        ("@+-", $alias_operand): $real_operands:
            When the operator is one of "@+-", this allows the user to use a
            single $alias_operand to specify one-or-more $real_operands to
            actually apply the operator to.
    """
    if not action: return d
    parts = re.split(r"(\+=|-=|\+\+|--|=|\+|-|@)", action, 1)
    attr, op, operand = ("", "+", parts[0]) if len(parts) == 1 else parts
    attr = attr.strip()
    target, target_one = rgetattr2(d, attr, one)

    if op in ("++", "--"):
        if operand:
            raise ValueError("action %s should have no operand: %s" % (action, operand))
    else:
        if not operand:
            raise ValueError("action %s has no operand" % action)

    operands = aliases.get(("@+-", operand), [operand]) if op in "@+-" else [operand]
    for operand in operands:
        if op == "@":
            # equivalent to - then +
            target = rdelattr(target, operand, rgetattr2(zero, attr)[0])
            target = rsetattr(target, operand, rgetattr2(target, operand, target_one)[0])
        elif op == "-":
            target = rdelattr(target, operand, rgetattr2(zero, attr)[0])
        elif op == "+":
            target = rsetattr(target, operand, rgetattr2(target, operand, target_one)[0])
        elif op == "=":
            if not isinstance(operand, target.__class__):
                operand = target.__class__(operand)
            target = operand
        elif op == "+=":
            if not isinstance(operand, target.__class__):
                operand = target.__class__(operand)
            target += operand
        elif op == "-=":
            if not isinstance(operand, target.__class__):
                operand = target.__class__(operand)
            target -= operand
        elif op == "++":
            target += 1
        elif op == "--":
            target -= 1
        else:
            assert False

    return rsetattr(d, attr, target, one) if attr else target

def parse_all(d, actions, one, zero=None, aliases={}, sep=None):
    """Parse a list of actions.

    If sep is given, each list element is further split on that separator.
    """
    if isinstance(actions, str):
        actions = [actions]
    if sep:
        actions = [a for aa in actions for a in aa.split(sep)]
    return functools.reduce(lambda d, a: parse(d, a, one, zero, aliases), actions, d)
