import copy
import uuid
from keystone.application_credential import schema as app_cred_schema
from keystone.assignment import schema as assignment_schema
from keystone.catalog import schema as catalog_schema
from keystone.common import validation
from keystone.common.validation import parameter_types
from keystone.common.validation import validators
from keystone.credential import schema as credential_schema
from keystone import exception
from keystone.federation import schema as federation_schema
from keystone.identity.backends import resource_options as ro
from keystone.identity import schema as identity_schema
from keystone.limit import schema as limit_schema
from keystone.oauth1 import schema as oauth1_schema
from keystone.policy import schema as policy_schema
from keystone.resource import schema as resource_schema
from keystone.tests import unit
from keystone.trust import schema as trust_schema
class IdentityProviderValidationTestCase(unit.BaseTestCase):
    """Test for V3 Identity Provider API validation."""

    def setUp(self):
        super(IdentityProviderValidationTestCase, self).setUp()
        create = federation_schema.identity_provider_create
        update = federation_schema.identity_provider_update
        self.create_idp_validator = validators.SchemaValidator(create)
        self.update_idp_validator = validators.SchemaValidator(update)

    def test_validate_idp_request_succeeds(self):
        """Test that we validate an identity provider request."""
        request_to_validate = {'description': 'identity provider description', 'enabled': True, 'remote_ids': [uuid.uuid4().hex, uuid.uuid4().hex]}
        self.create_idp_validator.validate(request_to_validate)
        self.update_idp_validator.validate(request_to_validate)

    def test_validate_idp_request_fails_with_invalid_params(self):
        """Exception raised when unknown parameter is found."""
        request_to_validate = {'bogus': uuid.uuid4().hex}
        self.assertRaises(exception.SchemaValidationError, self.create_idp_validator.validate, request_to_validate)
        self.assertRaises(exception.SchemaValidationError, self.update_idp_validator.validate, request_to_validate)

    def test_validate_idp_request_with_enabled(self):
        """Validate `enabled` as boolean-like values."""
        for valid_enabled in _VALID_ENABLED_FORMATS:
            request_to_validate = {'enabled': valid_enabled}
            self.create_idp_validator.validate(request_to_validate)
            self.update_idp_validator.validate(request_to_validate)

    def test_validate_idp_request_with_invalid_enabled_fails(self):
        """Exception is raised when `enabled` isn't a boolean-like value."""
        for invalid_enabled in _INVALID_ENABLED_FORMATS:
            request_to_validate = {'enabled': invalid_enabled}
            self.assertRaises(exception.SchemaValidationError, self.create_idp_validator.validate, request_to_validate)
            self.assertRaises(exception.SchemaValidationError, self.update_idp_validator.validate, request_to_validate)

    def test_validate_idp_request_no_parameters(self):
        """Test that schema validation with empty request body."""
        request_to_validate = {}
        self.create_idp_validator.validate(request_to_validate)
        self.assertRaises(exception.SchemaValidationError, self.update_idp_validator.validate, request_to_validate)

    def test_validate_idp_request_with_invalid_description_fails(self):
        """Exception is raised when `description` as a non-string value."""
        request_to_validate = {'description': False}
        self.assertRaises(exception.SchemaValidationError, self.create_idp_validator.validate, request_to_validate)
        self.assertRaises(exception.SchemaValidationError, self.update_idp_validator.validate, request_to_validate)

    def test_validate_idp_request_with_invalid_remote_id_fails(self):
        """Exception is raised when `remote_ids` is not a array."""
        request_to_validate = {'remote_ids': uuid.uuid4().hex}
        self.assertRaises(exception.SchemaValidationError, self.create_idp_validator.validate, request_to_validate)
        self.assertRaises(exception.SchemaValidationError, self.update_idp_validator.validate, request_to_validate)

    def test_validate_idp_request_with_duplicated_remote_id(self):
        """Exception is raised when the duplicated `remote_ids` is found."""
        idp_id = uuid.uuid4().hex
        request_to_validate = {'remote_ids': [idp_id, idp_id]}
        self.assertRaises(exception.SchemaValidationError, self.create_idp_validator.validate, request_to_validate)
        self.assertRaises(exception.SchemaValidationError, self.update_idp_validator.validate, request_to_validate)

    def test_validate_idp_request_remote_id_nullable(self):
        """Test that `remote_ids` could be explicitly set to None."""
        request_to_validate = {'remote_ids': None}
        self.create_idp_validator.validate(request_to_validate)
        self.update_idp_validator.validate(request_to_validate)