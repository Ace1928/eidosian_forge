from __future__ import unicode_literals
from pybtex.style.names import BaseNameStyle, name_part
from pybtex.style.template import join

        Format names similarly to {vv~}{ll}{, jj}{, f.} in BibTeX.

        >>> from pybtex.database import Person
        >>> name = Person(string=r"Charles Louis Xavier Joseph de la Vall{\'e}e Poussin")
        >>> lastfirst = NameStyle().format

        >>> print(lastfirst(name).format().render_as('latex'))
        de~la Vall{é}e~Poussin, Charles Louis Xavier~Joseph
        >>> print(lastfirst(name).format().render_as('html'))
        de&nbsp;la Vall<span class="bibtex-protected">é</span>e&nbsp;Poussin, Charles Louis Xavier&nbsp;Joseph

        >>> print(lastfirst(name, abbr=True).format().render_as('latex'))
        de~la Vall{é}e~Poussin, C.~L. X.~J.
        >>> print(lastfirst(name, abbr=True).format().render_as('html'))
        de&nbsp;la Vall<span class="bibtex-protected">é</span>e&nbsp;Poussin, C.&nbsp;L. X.&nbsp;J.

        >>> name = Person(first='First', last='Last', middle='Middle')
        >>> print(lastfirst(name).format().render_as('latex'))
        Last, First~Middle
        >>> print(lastfirst(name, abbr=True).format().render_as('latex'))
        Last, F.~M.

        