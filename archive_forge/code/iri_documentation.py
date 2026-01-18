from collections import namedtuple
from . import compat
from . import exceptions
from . import misc
from . import normalizers
from . import uri
Encode an IRIReference into a URIReference instance.

        If the ``idna`` module is installed or the ``rfc3986[idna]``
        extra is used then unicode characters in the IRI host
        component will be encoded with IDNA2008.

        :param idna_encoder:
            Function that encodes each part of the host component
            If not given will raise an exception if the IRI
            contains a host component.
        :rtype: uri.URIReference
        :returns: A URI reference
        