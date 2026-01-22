from typing import cast
from urllib.parse import quote as urlquote, unquote as urlunquote, urlunsplit
from hyperlink import URL as _URL
def _rereconstituter(name):
    """
    Attriute declaration to preserve mutability on L{URLPath}.

    @param name: a public attribute name
    @type name: native L{str}

    @return: a descriptor which retrieves the private version of the attribute
        on get and calls rerealize on set.
    """
    privateName = '_' + name
    return property(lambda self: getattr(self, privateName), lambda self, value: setattr(self, privateName, value if isinstance(value, bytes) else value.encode('charmap')) or self._reconstitute())