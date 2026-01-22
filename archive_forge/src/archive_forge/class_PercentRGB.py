import typing
class PercentRGB(typing.NamedTuple):
    """
    :class:`~typing.NamedTuple` representing a percentage RGB
    triplet.

    Has three fields, each of type :class:`str` and representing a
    percentage value in the range 0%-100% inclusive:

    .. attribute:: red

       The red portion of the color value.

    .. attribute:: green

       The green portion of the color value.

    .. attribute:: blue

       The blue portion of the color value.

    """
    red: str
    green: str
    blue: str