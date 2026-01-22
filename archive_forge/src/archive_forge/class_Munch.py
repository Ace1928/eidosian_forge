from .python3_compat import iterkeys, iteritems, Mapping  #, u
class Munch(dict):
    """ A dictionary that provides attribute-style access.

        >>> b = Munch()
        >>> b.hello = 'world'
        >>> b.hello
        'world'
        >>> b['hello'] += "!"
        >>> b.hello
        'world!'
        >>> b.foo = Munch(lol=True)
        >>> b.foo.lol
        True
        >>> b.foo is b['foo']
        True

        A Munch is a subclass of dict; it supports all the methods a dict does...

        >>> sorted(b.keys())
        ['foo', 'hello']

        Including update()...

        >>> b.update({ 'ponies': 'are pretty!' }, hello=42)
        >>> print (repr(b))
        Munch({'ponies': 'are pretty!', 'foo': Munch({'lol': True}), 'hello': 42})

        As well as iteration...

        >>> sorted([ (k,b[k]) for k in b ])
        [('foo', Munch({'lol': True})), ('hello', 42), ('ponies', 'are pretty!')]

        And "splats".

        >>> "The {knights} who say {ni}!".format(**Munch(knights='lolcats', ni='can haz'))
        'The lolcats who say can haz!'

        See unmunchify/Munch.toDict, munchify/Munch.fromDict for notes about conversion.
    """

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __getattr__(self, k):
        """ Gets key if it exists, otherwise throws AttributeError.

            nb. __getattr__ is only called if key is not found in normal places.

            >>> b = Munch(bar='baz', lol={})
            >>> b.foo
            Traceback (most recent call last):
                ...
            AttributeError: foo

            >>> b.bar
            'baz'
            >>> getattr(b, 'bar')
            'baz'
            >>> b['bar']
            'baz'

            >>> b.lol is b['lol']
            True
            >>> b.lol is getattr(b, 'lol')
            True
        """
        try:
            return object.__getattribute__(self, k)
        except AttributeError:
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

    def __setattr__(self, k, v):
        """ Sets attribute k if it exists, otherwise sets key k. A KeyError
            raised by set-item (only likely if you subclass Munch) will
            propagate as an AttributeError instead.

            >>> b = Munch(foo='bar', this_is='useful when subclassing')
            >>> hasattr(b.values, '__call__')
            True
            >>> b.values = 'uh oh'
            >>> b.values
            'uh oh'
            >>> b['values']
            Traceback (most recent call last):
                ...
            KeyError: 'values'
        """
        try:
            object.__getattribute__(self, k)
        except AttributeError:
            try:
                self[k] = v
            except:
                raise AttributeError(k)
        else:
            object.__setattr__(self, k, v)

    def __delattr__(self, k):
        """ Deletes attribute k if it exists, otherwise deletes key k. A KeyError
            raised by deleting the key--such as when the key is missing--will
            propagate as an AttributeError instead.

            >>> b = Munch(lol=42)
            >>> del b.lol
            >>> b.lol
            Traceback (most recent call last):
                ...
            AttributeError: lol
        """
        try:
            object.__getattribute__(self, k)
        except AttributeError:
            try:
                del self[k]
            except KeyError:
                raise AttributeError(k)
        else:
            object.__delattr__(self, k)

    def toDict(self):
        """ Recursively converts a munch back into a dictionary.

            >>> b = Munch(foo=Munch(lol=True), hello=42, ponies='are pretty!')
            >>> sorted(b.toDict().items())
            [('foo', {'lol': True}), ('hello', 42), ('ponies', 'are pretty!')]

            See unmunchify for more info.
        """
        return unmunchify(self)

    @property
    def __dict__(self):
        return self.toDict()

    def __repr__(self):
        """ Invertible* string-form of a Munch.

            >>> b = Munch(foo=Munch(lol=True), hello=42, ponies='are pretty!')
            >>> print (repr(b))
            Munch({'ponies': 'are pretty!', 'foo': Munch({'lol': True}), 'hello': 42})
            >>> eval(repr(b))
            Munch({'ponies': 'are pretty!', 'foo': Munch({'lol': True}), 'hello': 42})

            >>> with_spaces = Munch({1: 2, 'a b': 9, 'c': Munch({'simple': 5})})
            >>> print (repr(with_spaces))
            Munch({'a b': 9, 1: 2, 'c': Munch({'simple': 5})})
            >>> eval(repr(with_spaces))
            Munch({'a b': 9, 1: 2, 'c': Munch({'simple': 5})})

            (*) Invertible so long as collection contents are each repr-invertible.
        """
        return '{0}({1})'.format(self.__class__.__name__, dict.__repr__(self))

    def __dir__(self):
        return list(iterkeys(self))

    def __getstate__(self):
        """ Implement a serializable interface used for pickling.

        See https://docs.python.org/3.6/library/pickle.html.
        """
        return {k: v for k, v in self.items()}

    def __setstate__(self, state):
        """ Implement a serializable interface used for pickling.

        See https://docs.python.org/3.6/library/pickle.html.
        """
        self.clear()
        self.update(state)
    __members__ = __dir__

    @classmethod
    def fromDict(cls, d):
        """ Recursively transforms a dictionary into a Munch via copy.

            >>> b = Munch.fromDict({'urmom': {'sez': {'what': 'what'}}})
            >>> b.urmom.sez.what
            'what'

            See munchify for more info.
        """
        return munchify(d, cls)

    def copy(self):
        return type(self).fromDict(self)

    def update(self, *args, **kwargs):
        """
        Override built-in method to call custom __setitem__ method that may
        be defined in subclasses.
        """
        for k, v in iteritems(dict(*args, **kwargs)):
            self[k] = v

    def get(self, k, d=None):
        """
        D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.
        """
        if k not in self:
            return d
        return self[k]

    def setdefault(self, k, d=None):
        """
        D.setdefault(k[,d]) -> D.get(k,d), also set D[k]=d if k not in D
        """
        if k not in self:
            self[k] = d
        return self[k]