from .python3_compat import iterkeys, iteritems, Mapping  #, u
class DefaultFactoryMunch(Munch):
    """ A Munch that calls a user-specified function to generate values for
        missing keys like collections.defaultdict.

        >>> b = DefaultFactoryMunch(list, {'hello': 'world!'})
        >>> b.hello
        'world!'
        >>> b.foo
        []
        >>> b.bar.append('hello')
        >>> b.bar
        ['hello']
    """

    def __init__(self, default_factory, *args, **kwargs):
        super(DefaultFactoryMunch, self).__init__(*args, **kwargs)
        self.default_factory = default_factory

    @classmethod
    def fromDict(cls, d, default_factory):
        return munchify(d, factory=lambda d_: cls(default_factory, d_))

    def copy(self):
        return type(self).fromDict(self, default_factory=self.default_factory)

    def __repr__(self):
        factory = self.default_factory.__name__
        return '{0}({1}, {2})'.format(type(self).__name__, factory, dict.__repr__(self))

    def __setattr__(self, k, v):
        if k == 'default_factory':
            object.__setattr__(self, k, v)
        else:
            super(DefaultFactoryMunch, self).__setattr__(k, v)

    def __missing__(self, k):
        self[k] = self.default_factory()
        return self[k]