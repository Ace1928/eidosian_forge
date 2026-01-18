from .python3_compat import iterkeys, iteritems, Mapping  #, u
def unmunchify(x):
    """ Recursively converts a Munch into a dictionary.

        >>> b = Munch(foo=Munch(lol=True), hello=42, ponies='are pretty!')
        >>> sorted(unmunchify(b).items())
        [('foo', {'lol': True}), ('hello', 42), ('ponies', 'are pretty!')]

        unmunchify will handle intermediary dicts, lists and tuples (as well as
        their subclasses), but ymmv on custom datatypes.

        >>> b = Munch(foo=['bar', Munch(lol=True)], hello=42,
        ...         ponies=('are pretty!', Munch(lies='are trouble!')))
        >>> sorted(unmunchify(b).items()) #doctest: +NORMALIZE_WHITESPACE
        [('foo', ['bar', {'lol': True}]), ('hello', 42), ('ponies', ('are pretty!', {'lies': 'are trouble!'}))]

        nb. As dicts are not hashable, they cannot be nested in sets/frozensets.
    """
    seen = dict()

    def unmunchify_cycles(obj):
        try:
            return seen[id(obj)]
        except KeyError:
            pass
        seen[id(obj)] = partial = pre_unmunchify(obj)
        return post_unmunchify(partial, obj)

    def pre_unmunchify(obj):
        if isinstance(obj, Mapping):
            return dict()
        elif isinstance(obj, list):
            return type(obj)()
        elif isinstance(obj, tuple):
            type_factory = getattr(obj, '_make', type(obj))
            return type_factory((unmunchify_cycles(item) for item in obj))
        else:
            return obj

    def post_unmunchify(partial, obj):
        if isinstance(obj, Mapping):
            partial.update(((k, unmunchify_cycles(obj[k])) for k in iterkeys(obj)))
        elif isinstance(obj, list):
            partial.extend((unmunchify_cycles(v) for v in obj))
        elif isinstance(obj, tuple):
            for value_partial, value in zip(partial, obj):
                post_unmunchify(value_partial, value)
        return partial
    return unmunchify_cycles(x)