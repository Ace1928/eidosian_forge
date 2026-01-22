import binascii
import unicodedata
import base64
import cherrypy
from cherrypy._cpcompat import ntou, tonative
A CherryPy tool which hooks at before_handler to perform
    HTTP Basic Access Authentication, as specified in :rfc:`2617`
    and :rfc:`7617`.

    If the request has an 'authorization' header with a 'Basic' scheme, this
    tool attempts to authenticate the credentials supplied in that header.  If
    the request has no 'authorization' header, or if it does but the scheme is
    not 'Basic', or if authentication fails, the tool sends a 401 response with
    a 'WWW-Authenticate' Basic header.

    realm
        A string containing the authentication realm.

    checkpassword
        A callable which checks the authentication credentials.
        Its signature is checkpassword(realm, username, password). where
        username and password are the values obtained from the request's
        'authorization' header.  If authentication succeeds, checkpassword
        returns True, else it returns False.

    