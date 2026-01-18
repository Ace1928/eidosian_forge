import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from . import DistlibException
from .compat import (HTTPBasicAuthHandler, Request, HTTPPasswordMgr,
from .util import zip_dir, ServerProxy
def upload_documentation(self, metadata, doc_dir):
    """
        Upload documentation to the index.

        :param metadata: A :class:`Metadata` instance defining at least a name
                         and version number for the documentation to be
                         uploaded.
        :param doc_dir: The pathname of the directory which contains the
                        documentation. This should be the directory that
                        contains the ``index.html`` for the documentation.
        :return: The HTTP response received from PyPI upon submission of the
                request.
        """
    self.check_credentials()
    if not os.path.isdir(doc_dir):
        raise DistlibException('not a directory: %r' % doc_dir)
    fn = os.path.join(doc_dir, 'index.html')
    if not os.path.exists(fn):
        raise DistlibException('not found: %r' % fn)
    metadata.validate()
    name, version = (metadata.name, metadata.version)
    zip_data = zip_dir(doc_dir).getvalue()
    fields = [(':action', 'doc_upload'), ('name', name), ('version', version)]
    files = [('content', name, zip_data)]
    request = self.encode_request(fields, files)
    return self.send_request(request)