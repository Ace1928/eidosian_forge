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
def visititems_links(self, func):
    """ Recursively visit links in this group.
        Each link will be visited exactly once, regardless of its target.

        You supply a callable (function, method or callable object); it
        will be called exactly once for each link in this group and every
        group below it. Your callable must conform to the signature:

            func(<member name>, <link>) => <None or return value>

        Returning None continues iteration, returning anything else stops
        and immediately returns that value from the visit method.  No
        particular order of iteration within groups is guaranteed.

        Example:

        # Get a list of all softlinks in the file
        >>> mylist = []
        >>> def func(name, link):
        ...     if isinstance(link, SoftLink):
        ...         mylist.append(name)
        ...
        >>> f = File('foo.hdf5')
        >>> f.visititems_links(func)
        """
    with phil:

        def proxy(name):
            """ Use the text name of the object, not bytes """
            name = self._d(name)
            return func(name, self.get(name, getlink=True))
        return self.id.links.visit(proxy)