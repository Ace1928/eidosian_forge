from __future__ import print_function
from patsy import PatsyError
from patsy.origin import Origin
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
A token with possible payload.

    .. attribute:: type

       An arbitrary object indicating the type of this token. Should be
      :term:`hashable`, but otherwise it can be whatever you like.
    