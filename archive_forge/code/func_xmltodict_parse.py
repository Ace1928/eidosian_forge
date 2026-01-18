import json
import time
from typing import Any, Dict
import xmltodict
from blobfile import _xml as xml
def xmltodict_parse(data: bytes) -> Dict[str, Any]:
    parsed = xmltodict.parse(data.decode('utf8'))
    return remove_attributes(parsed)