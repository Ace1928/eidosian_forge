from urllib.parse import urlencode
from urllib.request import urlopen
from paste.request import construct_url
from paste.httpexceptions import HTTPSeeOther, HTTPForbidden
class CASAuthenticate(HTTPSeeOther):
    """ The exception raised to authenticate the user """