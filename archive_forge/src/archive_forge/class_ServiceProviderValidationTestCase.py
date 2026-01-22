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
class ServiceProviderValidationTestCase(unit.BaseTestCase):
    """Test for V3 Service Provider API validation."""

    def setUp(self):
        super(ServiceProviderValidationTestCase, self).setUp()
        self.valid_auth_url = 'https://' + uuid.uuid4().hex + '.com'
        self.valid_sp_url = 'https://' + uuid.uuid4().hex + '.com'
        create = federation_schema.service_provider_create
        update = federation_schema.service_provider_update
        self.create_sp_validator = validators.SchemaValidator(create)
        self.update_sp_validator = validators.SchemaValidator(update)

    def test_validate_sp_request(self):
        """Test that we validate `auth_url` and `sp_url` in request."""
        request_to_validate = {'auth_url': self.valid_auth_url, 'sp_url': self.valid_sp_url}
        self.create_sp_validator.validate(request_to_validate)

    def test_validate_sp_request_with_invalid_auth_url_fails(self):
        """Validate request fails with invalid `auth_url`."""
        request_to_validate = {'auth_url': uuid.uuid4().hex, 'sp_url': self.valid_sp_url}
        self.assertRaises(exception.SchemaValidationError, self.create_sp_validator.validate, request_to_validate)

    def test_validate_sp_request_with_invalid_sp_url_fails(self):
        """Validate request fails with invalid `sp_url`."""
        request_to_validate = {'auth_url': self.valid_auth_url, 'sp_url': uuid.uuid4().hex}
        self.assertRaises(exception.SchemaValidationError, self.create_sp_validator.validate, request_to_validate)

    def test_validate_sp_request_without_auth_url_fails(self):
        """Validate request fails without `auth_url`."""
        request_to_validate = {'sp_url': self.valid_sp_url}
        self.assertRaises(exception.SchemaValidationError, self.create_sp_validator.validate, request_to_validate)
        request_to_validate = {'auth_url': None, 'sp_url': self.valid_sp_url}
        self.assertRaises(exception.SchemaValidationError, self.create_sp_validator.validate, request_to_validate)

    def test_validate_sp_request_without_sp_url_fails(self):
        """Validate request fails without `sp_url`."""
        request_to_validate = {'auth_url': self.valid_auth_url}
        self.assertRaises(exception.SchemaValidationError, self.create_sp_validator.validate, request_to_validate)
        request_to_validate = {'auth_url': self.valid_auth_url, 'sp_url': None}
        self.assertRaises(exception.SchemaValidationError, self.create_sp_validator.validate, request_to_validate)

    def test_validate_sp_request_with_enabled(self):
        """Validate `enabled` as boolean-like values."""
        for valid_enabled in _VALID_ENABLED_FORMATS:
            request_to_validate = {'auth_url': self.valid_auth_url, 'sp_url': self.valid_sp_url, 'enabled': valid_enabled}
            self.create_sp_validator.validate(request_to_validate)

    def test_validate_sp_request_with_invalid_enabled_fails(self):
        """Exception is raised when `enabled` isn't a boolean-like value."""
        for invalid_enabled in _INVALID_ENABLED_FORMATS:
            request_to_validate = {'auth_url': self.valid_auth_url, 'sp_url': self.valid_sp_url, 'enabled': invalid_enabled}
            self.assertRaises(exception.SchemaValidationError, self.create_sp_validator.validate, request_to_validate)

    def test_validate_sp_request_with_valid_description(self):
        """Test that we validate `description` in create requests."""
        request_to_validate = {'auth_url': self.valid_auth_url, 'sp_url': self.valid_sp_url, 'description': 'My Service Provider'}
        self.create_sp_validator.validate(request_to_validate)

    def test_validate_sp_request_with_invalid_description_fails(self):
        """Exception is raised when `description` as a non-string value."""
        request_to_validate = {'auth_url': self.valid_auth_url, 'sp_url': self.valid_sp_url, 'description': False}
        self.assertRaises(exception.SchemaValidationError, self.create_sp_validator.validate, request_to_validate)

    def test_validate_sp_request_with_extra_field_fails(self):
        """Exception raised when passing extra fields in the body."""
        request_to_validate = {'id': 'ACME', 'auth_url': self.valid_auth_url, 'sp_url': self.valid_sp_url, 'description': 'My Service Provider'}
        self.assertRaises(exception.SchemaValidationError, self.create_sp_validator.validate, request_to_validate)

    def test_validate_sp_update_request(self):
        """Test that we validate a update request."""
        request_to_validate = {'description': uuid.uuid4().hex}
        self.update_sp_validator.validate(request_to_validate)

    def test_validate_sp_update_request_with_no_parameters_fails(self):
        """Exception is raised when updating without parameters."""
        request_to_validate = {}
        self.assertRaises(exception.SchemaValidationError, self.update_sp_validator.validate, request_to_validate)

    def test_validate_sp_update_request_with_invalid_auth_url_fails(self):
        """Exception raised when updating with invalid `auth_url`."""
        request_to_validate = {'auth_url': uuid.uuid4().hex}
        self.assertRaises(exception.SchemaValidationError, self.update_sp_validator.validate, request_to_validate)
        request_to_validate = {'auth_url': None}
        self.assertRaises(exception.SchemaValidationError, self.update_sp_validator.validate, request_to_validate)

    def test_validate_sp_update_request_with_invalid_sp_url_fails(self):
        """Exception raised when updating with invalid `sp_url`."""
        request_to_validate = {'sp_url': uuid.uuid4().hex}
        self.assertRaises(exception.SchemaValidationError, self.update_sp_validator.validate, request_to_validate)
        request_to_validate = {'sp_url': None}
        self.assertRaises(exception.SchemaValidationError, self.update_sp_validator.validate, request_to_validate)