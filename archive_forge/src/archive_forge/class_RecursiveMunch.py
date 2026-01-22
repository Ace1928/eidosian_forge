from collections.abc import Mapping
class RecursiveMunch(DefaultFactoryMunch):
    """A Munch that calls an instance of itself to generate values for
        missing keys.

        >>> b = RecursiveMunch({'hello': 'world!'})
        >>> b.hello
        'world!'
        >>> b.foo
        RecursiveMunch(RecursiveMunch, {})
        >>> b.bar.okay = 'hello'
        >>> b.bar
        RecursiveMunch(RecursiveMunch, {'okay': 'hello'})
        >>> b
        RecursiveMunch(RecursiveMunch, {'hello': 'world!', 'foo': RecursiveMunch(RecursiveMunch, {}),
        'bar': RecursiveMunch(RecursiveMunch, {'okay': 'hello'})})
    """

    def __init__(self, *args, **kwargs):
        super().__init__(RecursiveMunch, *args, **kwargs)

    @classmethod
    def fromDict(cls, d):
        return munchify(d, factory=cls)

    def copy(self):
        return type(self).fromDict(self)