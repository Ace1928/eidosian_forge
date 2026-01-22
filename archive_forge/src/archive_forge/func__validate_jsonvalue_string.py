import decimal
import json
from datetime import datetime
from botocore.exceptions import ParamValidationError
from botocore.utils import is_json_value_header, parse_to_aware_datetime
def _validate_jsonvalue_string(self, params, shape, errors, name):
    try:
        json.dumps(params)
    except (ValueError, TypeError) as e:
        errors.report(name, 'unable to encode to json', type_error=e)