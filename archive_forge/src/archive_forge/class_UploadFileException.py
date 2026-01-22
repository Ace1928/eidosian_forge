import os
from io import BytesIO
from django.conf import settings
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
from django.utils.module_loading import import_string
class UploadFileException(Exception):
    """
    Any error having to do with uploading files.
    """
    pass