import re
from nltk.tokenize.api import TokenizerI

        Return a list of s-expressions extracted from *text*.
        For example:

            >>> SExprTokenizer().tokenize('(a b (c d)) e f (g)')
            ['(a b (c d))', 'e', 'f', '(g)']

        All parentheses are assumed to mark s-expressions.
        (No special processing is done to exclude parentheses that occur
        inside strings, or following backslash characters.)

        If the given expression contains non-matching parentheses,
        then the behavior of the tokenizer depends on the ``strict``
        parameter to the constructor.  If ``strict`` is ``True``, then
        raise a ``ValueError``.  If ``strict`` is ``False``, then any
        unmatched close parentheses will be listed as their own
        s-expression; and the last partial s-expression with unmatched open
        parentheses will be listed as its own s-expression:

            >>> SExprTokenizer(strict=False).tokenize('c) d) e (f (g')
            ['c', ')', 'd', ')', 'e', '(f (g']

        :param text: the string to be tokenized
        :type text: str or iter(str)
        :rtype: iter(str)
        