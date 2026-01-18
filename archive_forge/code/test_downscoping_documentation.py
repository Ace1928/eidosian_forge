import re
import uuid
import google.auth
from google.auth import downscoped
from google.auth.transport import requests
from google.cloud import exceptions
from google.cloud import storage
from google.oauth2 import credentials
import pytest
Tests token consumer access to cloud storage using downscoped tokens.

    Args:
        temp_blobs (Tuple[google.cloud.storage.blob.Blob, ...]): The temporarily
            created test cloud storage blobs (one readonly accessible, the other
            not).
    