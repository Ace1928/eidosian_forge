import decimal
import json
from datetime import datetime
from botocore.exceptions import ParamValidationError
from botocore.utils import is_json_value_header, parse_to_aware_datetime
def _validate_blob(self, param, shape, errors, name):
    if isinstance(param, (bytes, bytearray, str)):
        return
    elif hasattr(param, 'read'):
        return
    else:
        errors.report(name, 'invalid type', param=param, valid_types=[str(bytes), str(bytearray), 'file-like object'])