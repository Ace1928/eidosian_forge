from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import utils
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from openstack import exceptions
from openstack.image.iterable_chunked_file import IterableChunkedFile
Image file signature generator.

    Generates signatures for files using a specified private key file.
    