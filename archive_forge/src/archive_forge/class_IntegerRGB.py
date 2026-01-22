import typing
class IntegerRGB(typing.NamedTuple):
    """
    :class:`~typing.NamedTuple` representing an integer RGB
    triplet.

    Has three fields, each of type :class:`int` and in the range 0-255
    inclusive:

    .. attribute:: red

       The red portion of the color value.

    .. attribute:: green

       The green portion of the color value.

    .. attribute:: blue

       The blue portion of the color value.

    """
    red: int
    green: int
    blue: int