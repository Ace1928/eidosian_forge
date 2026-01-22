import sys
import email.parser
from .encoder import encode_with
from requests.structures import CaseInsensitiveDict
def _split_on_find(content, bound):
    point = content.find(bound)
    return (content[:point], content[point + len(bound):])