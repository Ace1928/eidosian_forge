import base64
import calendar
import datetime
import json
import re
from xml.etree import ElementTree
from botocore import validate
from botocore.compat import formatdate
from botocore.exceptions import ParamValidationError
from botocore.utils import (
def _serialize_body_params(self, params, shape):
    root_name = shape.serialization['name']
    pseudo_root = ElementTree.Element('')
    self._serialize(shape, params, pseudo_root, root_name)
    real_root = list(pseudo_root)[0]
    return ElementTree.tostring(real_root, encoding=self.DEFAULT_ENCODING)