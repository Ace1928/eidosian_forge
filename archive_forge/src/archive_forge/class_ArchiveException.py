import os
import shutil
import stat
import tarfile
import zipfile
from django.core.exceptions import SuspiciousOperation
class ArchiveException(Exception):
    """
    Base exception class for all archive errors.
    """