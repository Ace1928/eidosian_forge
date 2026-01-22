import decimal
import json
from datetime import datetime
from botocore.exceptions import ParamValidationError
from botocore.utils import is_json_value_header, parse_to_aware_datetime
def _validate_document(self, params, shape, errors, name):
    if params is None:
        return
    if isinstance(params, dict):
        for key in params:
            self._validate_document(params[key], shape, errors, key)
    elif isinstance(params, list):
        for index, entity in enumerate(params):
            self._validate_document(entity, shape, errors, '%s[%d]' % (name, index))
    elif not isinstance(params, ((str,), int, bool, float)):
        valid_types = (str, int, bool, float, list, dict)
        valid_type_names = [str(t) for t in valid_types]
        errors.report(name, 'invalid type for document', param=params, param_type=type(params), valid_types=valid_type_names)