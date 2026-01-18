import os
from io import BytesIO
from django.conf import settings
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
from django.utils.module_loading import import_string
def upload_complete(self):
    """
        Signal that the upload is complete. Subclasses should perform cleanup
        that is necessary for this handler.
        """
    pass