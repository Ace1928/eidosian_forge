import json
import time
from typing import Any, Dict
import xmltodict
from blobfile import _xml as xml
def test_unparse():
    body = {'BlockList': {'Latest': [str(i) for i in range(100)]}}
    ref = xmltodict_unparse(body)
    actual = xml.unparse(body)
    print(ref)
    print(actual)
    assert ref == actual