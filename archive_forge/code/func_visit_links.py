from contextlib import contextmanager
import posixpath as pp
import numpy
from .compat import filename_decode, filename_encode
from .. import h5, h5g, h5i, h5o, h5r, h5t, h5l, h5p
from . import base
from .base import HLObject, MutableMappingHDF5, phil, with_phil
from . import dataset
from . import datatype
from .vds import vds_support
def visit_links(self, func):
    """ Recursively visit all names in this group and subgroups.
        Each link will be visited exactly once, regardless of its target.

        You supply a callable (function, method or callable object); it
        will be called exactly once for each link in this group and every
        group below it. Your callable must conform to the signature:

            func(<member name>) => <None or return value>

        Returning None continues iteration, returning anything else stops
        and immediately returns that value from the visit method.  No
        particular order of iteration within groups is guaranteed.

        Example:

        >>> # List the entire contents of the file
        >>> f = File("foo.hdf5")
        >>> list_of_names = []
        >>> f.visit_links(list_of_names.append)
        """
    with phil:

        def proxy(name):
            """ Call the function with the text name, not bytes """
            return func(self._d(name))
        return self.id.links.visit(proxy)