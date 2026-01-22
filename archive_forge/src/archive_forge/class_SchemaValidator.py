import re
import jsonschema
from oslo_config import cfg
from oslo_log import log
from keystone import exception
from keystone.i18n import _
class SchemaValidator(object):
    """Resource reference validator class."""
    validator_org = jsonschema.Draft4Validator

    def __init__(self, schema):
        validators = {}
        validator_cls = jsonschema.validators.extend(self.validator_org, validators)
        fc = jsonschema.FormatChecker()
        self.validator = validator_cls(schema, format_checker=fc)

    def validate(self, *args, **kwargs):
        try:
            self.validator.validate(*args, **kwargs)
        except jsonschema.ValidationError as ex:
            if ex.path:
                path = '/'.join(map(str, ex.path))
                detail = _("Invalid input for field '%(path)s': %(message)s") % {'path': path, 'message': str(ex)}
            else:
                detail = str(ex)
            raise exception.SchemaValidationError(detail=detail)