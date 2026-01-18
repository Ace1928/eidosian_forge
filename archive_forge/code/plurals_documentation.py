from __future__ import annotations
from operator import itemgetter
from babel.core import Locale, default_locale
A tuple with the information catalogs need to perform proper
    pluralization.  The first item of the tuple is the number of plural
    forms, the second the plural expression.

    >>> get_plural(locale='en')
    (2, '(n != 1)')
    >>> get_plural(locale='ga')
    (5, '(n==1 ? 0 : n==2 ? 1 : n>=3 && n<=6 ? 2 : n>=7 && n<=10 ? 3 : 4)')

    The object returned is a special tuple with additional members:

    >>> tup = get_plural("ja")
    >>> tup.num_plurals
    1
    >>> tup.plural_expr
    '0'
    >>> tup.plural_forms
    'nplurals=1; plural=0;'

    Converting the tuple into a string prints the plural forms for a
    gettext catalog:

    >>> str(tup)
    'nplurals=1; plural=0;'
    