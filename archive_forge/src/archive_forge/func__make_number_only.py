import re
def _make_number_only(self, ls, title, nc=(), s1=''):
    """Summarise number of cuts as a string (PRIVATE).

        Return a string of form::

            title.

            enzyme which cut 1 time:

            enzyme1     :   position1.

            enzyme which cut 2 times:

            enzyme2     :   position1, position2.
            ...

        Arguments:
         - ls is a list of results.
         - title is a string.
         - Non cutting enzymes are not included.
        """
    if not ls:
        return title
    ls.sort(key=lambda x: len(x[1]))
    iterator = iter(ls)
    cur_len = 1
    new_sect = []
    for name, sites in iterator:
        length = len(sites)
        if length > cur_len:
            title += '\n\nenzymes which cut %i times :\n\n' % cur_len
            title = self.__next_section(new_sect, title)
            new_sect, cur_len = ([(name, sites)], length)
            continue
        new_sect.append((name, sites))
    title += '\n\nenzymes which cut %i times :\n\n' % cur_len
    return self.__next_section(new_sect, title)