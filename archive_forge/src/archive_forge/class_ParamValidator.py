import decimal
import json
from datetime import datetime
from botocore.exceptions import ParamValidationError
from botocore.utils import is_json_value_header, parse_to_aware_datetime
class ParamValidator:
    """Validates parameters against a shape model."""

    def validate(self, params, shape):
        """Validate parameters against a shape model.

        This method will validate the parameters against a provided shape model.
        All errors will be collected before returning to the caller.  This means
        that this method will not stop at the first error, it will return all
        possible errors.

        :param params: User provided dict of parameters
        :param shape: A shape model describing the expected input.

        :return: A list of errors.

        """
        errors = ValidationErrors()
        self._validate(params, shape, errors, name='')
        return errors

    def _check_special_validation_cases(self, shape):
        if is_json_value_header(shape):
            return self._validate_jsonvalue_string
        if shape.type_name == 'structure' and shape.is_document_type:
            return self._validate_document

    def _validate(self, params, shape, errors, name):
        special_validator = self._check_special_validation_cases(shape)
        if special_validator:
            special_validator(params, shape, errors, name)
        else:
            getattr(self, '_validate_%s' % shape.type_name)(params, shape, errors, name)

    def _validate_jsonvalue_string(self, params, shape, errors, name):
        try:
            json.dumps(params)
        except (ValueError, TypeError) as e:
            errors.report(name, 'unable to encode to json', type_error=e)

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

    @type_check(valid_types=(dict,))
    def _validate_structure(self, params, shape, errors, name):
        if shape.is_tagged_union:
            if len(params) == 0:
                errors.report(name, 'empty input', members=shape.members)
            elif len(params) > 1:
                errors.report(name, 'more than one input', members=shape.members)
        for required_member in shape.metadata.get('required', []):
            if required_member not in params:
                errors.report(name, 'missing required field', required_name=required_member, user_params=params)
        members = shape.members
        known_params = []
        for param in params:
            if param not in members:
                errors.report(name, 'unknown field', unknown_param=param, valid_names=list(members))
            else:
                known_params.append(param)
        for param in known_params:
            self._validate(params[param], shape.members[param], errors, f'{name}.{param}')

    @type_check(valid_types=(str,))
    def _validate_string(self, param, shape, errors, name):
        range_check(name, len(param), shape, 'invalid length', errors)

    @type_check(valid_types=(list, tuple))
    def _validate_list(self, param, shape, errors, name):
        member_shape = shape.member
        range_check(name, len(param), shape, 'invalid length', errors)
        for i, item in enumerate(param):
            self._validate(item, member_shape, errors, f'{name}[{i}]')

    @type_check(valid_types=(dict,))
    def _validate_map(self, param, shape, errors, name):
        key_shape = shape.key
        value_shape = shape.value
        for key, value in param.items():
            self._validate(key, key_shape, errors, f'{name} (key: {key})')
            self._validate(value, value_shape, errors, f'{name}.{key}')

    @type_check(valid_types=(int,))
    def _validate_integer(self, param, shape, errors, name):
        range_check(name, param, shape, 'invalid range', errors)

    def _validate_blob(self, param, shape, errors, name):
        if isinstance(param, (bytes, bytearray, str)):
            return
        elif hasattr(param, 'read'):
            return
        else:
            errors.report(name, 'invalid type', param=param, valid_types=[str(bytes), str(bytearray), 'file-like object'])

    @type_check(valid_types=(bool,))
    def _validate_boolean(self, param, shape, errors, name):
        pass

    @type_check(valid_types=(float, decimal.Decimal) + (int,))
    def _validate_double(self, param, shape, errors, name):
        range_check(name, param, shape, 'invalid range', errors)
    _validate_float = _validate_double

    @type_check(valid_types=(int,))
    def _validate_long(self, param, shape, errors, name):
        range_check(name, param, shape, 'invalid range', errors)

    def _validate_timestamp(self, param, shape, errors, name):
        is_valid_type = self._type_check_datetime(param)
        if not is_valid_type:
            valid_type_names = [str(datetime), 'timestamp-string']
            errors.report(name, 'invalid type', param=param, valid_types=valid_type_names)

    def _type_check_datetime(self, value):
        try:
            parse_to_aware_datetime(value)
            return True
        except (TypeError, ValueError, AttributeError):
            return False