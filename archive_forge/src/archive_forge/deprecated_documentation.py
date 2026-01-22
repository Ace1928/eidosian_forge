import re
import sys
from requests import utils
Return the requested content back in unicode.

    This will first attempt to retrieve the encoding from the response
    headers. If that fails, it will use
    :func:`requests_toolbelt.utils.deprecated.get_encodings_from_content`
    to determine encodings from HTML elements.

    .. code-block:: python

        import requests
        from requests_toolbelt.utils import deprecated

        r = requests.get(url)
        text = deprecated.get_unicode_from_response(r)

    :param response: Response object to get unicode content from.
    :type response: requests.models.Response
    