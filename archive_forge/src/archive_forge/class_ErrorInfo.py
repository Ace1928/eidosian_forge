import json
from collections import namedtuple
import macaroonbakery.bakery as bakery
class ErrorInfo(namedtuple('ErrorInfo', 'macaroon, macaroon_path, cookie_name_suffix, interaction_methods, visit_url, wait_url')):
    """  Holds additional information provided
    by an error.

    @param macaroon may hold a macaroon that, when
    discharged, may allow access to a service.
    This field is associated with the ERR_DISCHARGE_REQUIRED
    error code.

    @param macaroon_path holds the URL path to be associated
    with the macaroon. The macaroon is potentially
    valid for all URLs under the given path.
    If it is empty, the macaroon will be associated with
    the original URL from which the error was returned.

    @param cookie_name_suffix holds the desired cookie name suffix to be
    associated with the macaroon. The actual name used will be
    ("macaroon-" + cookie_name_suffix). Clients may ignore this field -
    older clients will always use ("macaroon-" +
    macaroon.signature() in hex).

    @param visit_url holds a URL that the client should visit
    in a web browser to authenticate themselves.

    @param wait_url holds a URL that the client should visit
    to acquire the discharge macaroon. A GET on
    this URL will block until the client has authenticated,
    and then it will return the discharge macaroon.
    """
    __slots__ = ()

    @classmethod
    def from_dict(cls, serialized):
        """Create a new ErrorInfo object from a JSON deserialized
        dictionary
        @param serialized The JSON object {dict}
        @return ErrorInfo object
        """
        if serialized is None:
            return None
        macaroon = serialized.get('Macaroon')
        if macaroon is not None:
            macaroon = bakery.Macaroon.from_dict(macaroon)
        path = serialized.get('MacaroonPath')
        cookie_name_suffix = serialized.get('CookieNameSuffix')
        visit_url = serialized.get('VisitURL')
        wait_url = serialized.get('WaitURL')
        interaction_methods = serialized.get('InteractionMethods')
        return ErrorInfo(macaroon=macaroon, macaroon_path=path, cookie_name_suffix=cookie_name_suffix, visit_url=visit_url, wait_url=wait_url, interaction_methods=interaction_methods)

    def __new__(cls, macaroon=None, macaroon_path=None, cookie_name_suffix=None, interaction_methods=None, visit_url=None, wait_url=None):
        """Override the __new__ method so that we can
        have optional arguments, which namedtuple doesn't
        allow"""
        return super(ErrorInfo, cls).__new__(cls, macaroon, macaroon_path, cookie_name_suffix, interaction_methods, visit_url, wait_url)