from __future__ import annotations
from typing import Any, Dict
Merge two Structs with customizable conflict resolution.

        This is similar to :meth:`update`, but much more flexible. First, a
        dict is made from data+key=value pairs. When merging this dict with
        the Struct S, the optional dictionary 'conflict' is used to decide
        what to do.

        If conflict is not given, the default behavior is to preserve any keys
        with their current value (the opposite of the :meth:`update` method's
        behavior).

        Parameters
        ----------
        __loc_data__ : dict, Struct
            The data to merge into self
        __conflict_solve : dict
            The conflict policy dict.  The keys are binary functions used to
            resolve the conflict and the values are lists of strings naming
            the keys the conflict resolution function applies to.  Instead of
            a list of strings a space separated string can be used, like
            'a b c'.
        **kw : dict
            Additional key, value pairs to merge in

        Notes
        -----
        The `__conflict_solve` dict is a dictionary of binary functions which will be used to
        solve key conflicts.  Here is an example::

            __conflict_solve = dict(
                func1=['a','b','c'],
                func2=['d','e']
            )

        In this case, the function :func:`func1` will be used to resolve
        keys 'a', 'b' and 'c' and the function :func:`func2` will be used for
        keys 'd' and 'e'.  This could also be written as::

            __conflict_solve = dict(func1='a b c',func2='d e')

        These functions will be called for each key they apply to with the
        form::

            func1(self['a'], other['a'])

        The return value is used as the final merged value.

        As a convenience, merge() provides five (the most commonly needed)
        pre-defined policies: preserve, update, add, add_flip and add_s. The
        easiest explanation is their implementation::

            preserve = lambda old,new: old
            update   = lambda old,new: new
            add      = lambda old,new: old + new
            add_flip = lambda old,new: new + old  # note change of order!
            add_s    = lambda old,new: old + ' ' + new  # only for str!

        You can use those four words (as strings) as keys instead
        of defining them as functions, and the merge method will substitute
        the appropriate functions for you.

        For more complicated conflict resolution policies, you still need to
        construct your own functions.

        Examples
        --------
        This show the default policy:

        >>> s = Struct(a=10,b=30)
        >>> s2 = Struct(a=20,c=40)
        >>> s.merge(s2)
        >>> sorted(s.items())
        [('a', 10), ('b', 30), ('c', 40)]

        Now, show how to specify a conflict dict:

        >>> s = Struct(a=10,b=30)
        >>> s2 = Struct(a=20,b=40)
        >>> conflict = {'update':'a','add':'b'}
        >>> s.merge(s2,conflict)
        >>> sorted(s.items())
        [('a', 20), ('b', 70)]
        