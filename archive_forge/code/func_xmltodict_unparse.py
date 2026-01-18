import json
import time
from typing import Any, Dict
import xmltodict
from blobfile import _xml as xml
def xmltodict_unparse(d: Dict[str, Any]) -> bytes:
    return xmltodict.unparse(d).encode('utf8')