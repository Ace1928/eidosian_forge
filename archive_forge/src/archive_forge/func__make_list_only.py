import re
def _make_list_only(self, ls, title, nc=(), s1=''):
    """Summarise list of positions per enzyme (PRIVATE).

        Return a string of form::

            title.

            enzyme1     :   position1, position2.
            enzyme2     :   position1, position2, position3.
            ...

        Arguments:
         - ls is a tuple or list of results.
         - title is a string.
         - Non cutting enzymes are not included.
        """
    if not ls:
        return title
    return self.__next_section(ls, title)