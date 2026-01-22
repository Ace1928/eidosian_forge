from __future__ import absolute_import
import os
import re
import json
import mimetypes
import tempfile
from multiprocessing.pool import ThreadPool
from datetime import date, datetime
from six import PY3, integer_types, iteritems, text_type
from six.moves.urllib.parse import quote
from . import models
from .configuration import Configuration
from .rest import ApiException, RESTClientObject
def __deserialize_file(self, response):
    """
        Saves response body into a file in a temporary folder,
        using the filename from the `Content-Disposition` header if provided.

        :param response:  RESTResponse.
        :return: file path.
        """
    fd, path = tempfile.mkstemp(dir=self.configuration.temp_folder_path)
    os.close(fd)
    os.remove(path)
    content_disposition = response.getheader('Content-Disposition')
    if content_disposition:
        filename = re.search('filename=[\\\'"]?([^\\\'"\\s]+)[\\\'"]?', content_disposition).group(1)
        path = os.path.join(os.path.dirname(path), filename)
    with open(path, 'w') as f:
        f.write(response.data)
    return path