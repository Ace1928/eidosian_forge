import re
Next section (PRIVATE).

        Arguments:
         - ls is a tuple/list of tuple (string, [int, int]).
         - into is a string to which the formatted ls will be added.

        Format ls as a string of lines:
        The form is::

            enzyme1     :   position1.
            enzyme2     :   position2, position3.

        then add the formatted ls to tot
        return tot.
        