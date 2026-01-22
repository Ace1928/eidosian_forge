import sys
import email.parser
from .encoder import encode_with
from requests.structures import CaseInsensitiveDict
class NonMultipartContentTypeException(Exception):
    pass