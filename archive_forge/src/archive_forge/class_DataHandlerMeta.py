import os
import warnings
from collections import Counter
from xml.parsers import expat
from io import BytesIO
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from urllib.request import urlopen
from urllib.parse import urlparse
from Bio import StreamModeError
class DataHandlerMeta(type):
    """A metaclass is needed until Python supports @classproperty."""

    def __init__(cls, *args, **kwargs):
        """Initialize the class."""
        try:
            cls.directory = None
        except PermissionError:
            cls._directory = None

    @property
    def directory(cls):
        """Directory for caching XSD and DTD files."""
        return cls._directory

    @directory.setter
    def directory(cls, value):
        """Set a custom directory for the local DTD/XSD directories."""
        if value is None:
            import platform
            if platform.system() == 'Windows':
                value = os.path.join(os.getenv('APPDATA'), 'biopython')
            else:
                home = os.path.expanduser('~')
                value = os.path.join(home, '.config', 'biopython')
        cls.local_dtd_dir = os.path.join(value, 'Bio', 'Entrez', 'DTDs')
        os.makedirs(cls.local_dtd_dir, exist_ok=True)
        cls.local_xsd_dir = os.path.join(value, 'Bio', 'Entrez', 'XSDs')
        os.makedirs(cls.local_xsd_dir, exist_ok=True)
        cls._directory = value