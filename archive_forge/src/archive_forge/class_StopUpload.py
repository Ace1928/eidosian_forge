import os
from io import BytesIO
from django.conf import settings
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
from django.utils.module_loading import import_string
class StopUpload(UploadFileException):
    """
    This exception is raised when an upload must abort.
    """

    def __init__(self, connection_reset=False):
        """
        If ``connection_reset`` is ``True``, Django knows will halt the upload
        without consuming the rest of the upload. This will cause the browser to
        show a "connection reset" error.
        """
        self.connection_reset = connection_reset

    def __str__(self):
        if self.connection_reset:
            return 'StopUpload: Halt current upload.'
        else:
            return 'StopUpload: Consume request data, then halt.'