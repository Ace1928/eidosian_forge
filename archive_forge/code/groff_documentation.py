import math
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt

    Format tokens with groff escapes to change their color and font style.

    .. versionadded:: 2.11

    Additional options accepted:

    `style`
        The style to use, can be a string or a Style subclass (default:
        ``'default'``).

    `monospaced`
        If set to true, monospace font will be used (default: ``true``).

    `linenos`
        If set to true, print the line numbers (default: ``false``).

    `wrap`
        Wrap lines to the specified number of characters. Disabled if set to 0
        (default: ``0``).
    