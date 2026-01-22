from collections.abc import Sequence
from typing import Optional, Union, cast
from twisted.python.compat import nativeString
from twisted.web._responses import RESPONSES
class PageRedirect(Error):
    """
    A request resulted in an HTTP redirect.

    @ivar location: The location of the redirect which was not followed.
    """
    location: Optional[bytes]

    def __init__(self, code: Union[int, bytes], message: Optional[bytes]=None, response: Optional[bytes]=None, location: Optional[bytes]=None) -> None:
        """
        Initializes a page redirect exception.

        @type code: L{bytes}
        @param code: Refers to an HTTP status code, for example
            C{http.NOT_FOUND}. If no C{message} is given, C{code} is mapped to a
            descriptive string that is used instead.

        @type message: L{bytes}
        @param message: A short error message, for example C{b"NOT FOUND"}.

        @type response: L{bytes}
        @param response: A complete HTML document for an error page.

        @type location: L{bytes}
        @param location: The location response-header field value. It is an
            absolute URI used to redirect the receiver to a location other than
            the Request-URI so the request can be completed.
        """
        Error.__init__(self, code, message, response)
        if self.message and location:
            self.message = self.message + b' to ' + location
        self.location = location