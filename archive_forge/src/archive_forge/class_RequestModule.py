from __future__ import absolute_import
import sys
from .filepost import encode_multipart_formdata
from .packages import six
from .packages.six.moves.urllib.parse import urlencode
class RequestModule(sys.modules[__name__].__class__):

    def __call__(self, *args, **kwargs):
        """
            If user tries to call this module directly urllib3 v2.x style raise an error to the user
            suggesting they may need urllib3 v2
            """
        raise TypeError("'module' object is not callable\nurllib3.request() method is not supported in this release, upgrade to urllib3 v2 to use it\nsee https://urllib3.readthedocs.io/en/stable/v2-migration-guide.html")