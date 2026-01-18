import typing
Wrap typing.overload that remembers the overloaded signatures

    This provides a custom implementation of typing.overload that
    remembers the overloaded signatures so that they are available for
    runtime inspection.

    