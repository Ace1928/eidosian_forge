import datetime
import email
from io import StringIO
from urllib.parse import urlparse
from django.utils.encoding import iri_to_uri
from django.utils.xmlutils import SimplerXMLGenerator
class Enclosure:
    """An RSS enclosure"""

    def __init__(self, url, length, mime_type):
        """All args are expected to be strings"""
        self.length, self.mime_type = (length, mime_type)
        self.url = iri_to_uri(url)