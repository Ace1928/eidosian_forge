import re

    Replace the units string provided with an equivalent html string.

    Exponentiation (m**2) will be replaced with superscripts (m<sup>2</sup>})

    No formatting is done, change `font` argument to e.g.:
    '<span style="color: #0000a0">%s</span>' to have text be colored blue.

    Multiplication (*) are replaced with the symbol specified by the mult
    argument. By default this is the latex &sdot; symbol.  Other useful options
    may be '' or '*'.

    If paren=True, encapsulate the string in '(' and ')'

    