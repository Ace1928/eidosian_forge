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
class EndpointValidationTestCase(unit.BaseTestCase):
    """Test for V3 Endpoint API validation."""

    def setUp(self):
        super(EndpointValidationTestCase, self).setUp()
        create = catalog_schema.endpoint_create
        update = catalog_schema.endpoint_update
        self.create_endpoint_validator = validators.SchemaValidator(create)
        self.update_endpoint_validator = validators.SchemaValidator(update)

    def test_validate_endpoint_request_succeeds(self):
        """Test that we validate an endpoint request."""
        request_to_validate = {'enabled': True, 'interface': 'admin', 'region_id': uuid.uuid4().hex, 'service_id': uuid.uuid4().hex, 'url': 'https://service.example.com:5000/'}
        self.create_endpoint_validator.validate(request_to_validate)

    def test_validate_endpoint_create_succeeds_with_required_parameters(self):
        """Validate an endpoint request with only the required parameters."""
        request_to_validate = {'service_id': uuid.uuid4().hex, 'interface': 'public', 'url': 'https://service.example.com:5000/'}
        self.create_endpoint_validator.validate(request_to_validate)

    def test_validate_endpoint_create_succeeds_with_valid_enabled(self):
        """Validate an endpoint with boolean values.

        Validate boolean values as `enabled` in endpoint create requests.
        """
        for valid_enabled in _VALID_ENABLED_FORMATS:
            request_to_validate = {'enabled': valid_enabled, 'service_id': uuid.uuid4().hex, 'interface': 'public', 'url': 'https://service.example.com:5000/'}
            self.create_endpoint_validator.validate(request_to_validate)

    def test_validate_create_endpoint_fails_with_invalid_enabled(self):
        """Exception raised when boolean-like values as `enabled`."""
        for invalid_enabled in _INVALID_ENABLED_FORMATS:
            request_to_validate = {'enabled': invalid_enabled, 'service_id': uuid.uuid4().hex, 'interface': 'public', 'url': 'https://service.example.com:5000/'}
            self.assertRaises(exception.SchemaValidationError, self.create_endpoint_validator.validate, request_to_validate)

    def test_validate_endpoint_create_succeeds_with_extra_parameters(self):
        """Test that extra parameters pass validation on create endpoint."""
        request_to_validate = {'other_attr': uuid.uuid4().hex, 'service_id': uuid.uuid4().hex, 'interface': 'public', 'url': 'https://service.example.com:5000/'}
        self.create_endpoint_validator.validate(request_to_validate)

    def test_validate_endpoint_create_fails_without_service_id(self):
        """Exception raised when `service_id` isn't in endpoint request."""
        request_to_validate = {'interface': 'public', 'url': 'https://service.example.com:5000/'}
        self.assertRaises(exception.SchemaValidationError, self.create_endpoint_validator.validate, request_to_validate)

    def test_validate_endpoint_create_fails_without_interface(self):
        """Exception raised when `interface` isn't in endpoint request."""
        request_to_validate = {'service_id': uuid.uuid4().hex, 'url': 'https://service.example.com:5000/'}
        self.assertRaises(exception.SchemaValidationError, self.create_endpoint_validator.validate, request_to_validate)

    def test_validate_endpoint_create_fails_without_url(self):
        """Exception raised when `url` isn't in endpoint request."""
        request_to_validate = {'service_id': uuid.uuid4().hex, 'interface': 'public'}
        self.assertRaises(exception.SchemaValidationError, self.create_endpoint_validator.validate, request_to_validate)

    def test_validate_endpoint_create_succeeds_with_url(self):
        """Validate `url` attribute in endpoint create request."""
        request_to_validate = {'service_id': uuid.uuid4().hex, 'interface': 'public'}
        for url in _VALID_URLS:
            request_to_validate['url'] = url
            self.create_endpoint_validator.validate(request_to_validate)

    def test_validate_endpoint_create_fails_with_invalid_url(self):
        """Exception raised when passing invalid `url` in request."""
        request_to_validate = {'service_id': uuid.uuid4().hex, 'interface': 'public'}
        for url in _INVALID_URLS:
            request_to_validate['url'] = url
            self.assertRaises(exception.SchemaValidationError, self.create_endpoint_validator.validate, request_to_validate)

    def test_validate_endpoint_create_fails_with_invalid_interface(self):
        """Exception raised with invalid `interface`."""
        request_to_validate = {'interface': uuid.uuid4().hex, 'service_id': uuid.uuid4().hex, 'url': 'https://service.example.com:5000/'}
        self.assertRaises(exception.SchemaValidationError, self.create_endpoint_validator.validate, request_to_validate)

    def test_validate_endpoint_create_fails_with_invalid_region_id(self):
        """Exception raised when passing invalid `region(_id)` in request."""
        request_to_validate = {'interface': 'admin', 'region_id': 1234, 'service_id': uuid.uuid4().hex, 'url': 'https://service.example.com:5000/'}
        self.assertRaises(exception.SchemaValidationError, self.create_endpoint_validator.validate, request_to_validate)
        request_to_validate = {'interface': 'admin', 'region': 1234, 'service_id': uuid.uuid4().hex, 'url': 'https://service.example.com:5000/'}
        self.assertRaises(exception.SchemaValidationError, self.create_endpoint_validator.validate, request_to_validate)

    def test_validate_endpoint_update_fails_with_invalid_enabled(self):
        """Exception raised when `enabled` is boolean-like value."""
        for invalid_enabled in _INVALID_ENABLED_FORMATS:
            request_to_validate = {'enabled': invalid_enabled}
            self.assertRaises(exception.SchemaValidationError, self.update_endpoint_validator.validate, request_to_validate)

    def test_validate_endpoint_update_succeeds_with_valid_enabled(self):
        """Validate `enabled` as boolean values."""
        for valid_enabled in _VALID_ENABLED_FORMATS:
            request_to_validate = {'enabled': valid_enabled}
            self.update_endpoint_validator.validate(request_to_validate)

    def test_validate_endpoint_update_fails_with_invalid_interface(self):
        """Exception raised when invalid `interface` on endpoint update."""
        request_to_validate = {'interface': uuid.uuid4().hex, 'service_id': uuid.uuid4().hex, 'url': 'https://service.example.com:5000/'}
        self.assertRaises(exception.SchemaValidationError, self.update_endpoint_validator.validate, request_to_validate)

    def test_validate_endpoint_update_request_succeeds(self):
        """Test that we validate an endpoint update request."""
        request_to_validate = {'enabled': True, 'interface': 'admin', 'region_id': uuid.uuid4().hex, 'service_id': uuid.uuid4().hex, 'url': 'https://service.example.com:5000/'}
        self.update_endpoint_validator.validate(request_to_validate)

    def test_validate_endpoint_update_fails_with_no_parameters(self):
        """Exception raised when no parameters on endpoint update."""
        request_to_validate = {}
        self.assertRaises(exception.SchemaValidationError, self.update_endpoint_validator.validate, request_to_validate)

    def test_validate_endpoint_update_succeeds_with_extra_parameters(self):
        """Test that extra parameters pass validation on update endpoint."""
        request_to_validate = {'enabled': True, 'interface': 'admin', 'region_id': uuid.uuid4().hex, 'service_id': uuid.uuid4().hex, 'url': 'https://service.example.com:5000/', 'other_attr': uuid.uuid4().hex}
        self.update_endpoint_validator.validate(request_to_validate)

    def test_validate_endpoint_update_succeeds_with_url(self):
        """Validate `url` attribute in endpoint update request."""
        request_to_validate = {'service_id': uuid.uuid4().hex, 'interface': 'public'}
        for url in _VALID_URLS:
            request_to_validate['url'] = url
            self.update_endpoint_validator.validate(request_to_validate)

    def test_validate_endpoint_update_fails_with_invalid_url(self):
        """Exception raised when passing invalid `url` in request."""
        request_to_validate = {'service_id': uuid.uuid4().hex, 'interface': 'public'}
        for url in _INVALID_URLS:
            request_to_validate['url'] = url
            self.assertRaises(exception.SchemaValidationError, self.update_endpoint_validator.validate, request_to_validate)

    def test_validate_endpoint_update_fails_with_invalid_region_id(self):
        """Exception raised when passing invalid `region(_id)` in request."""
        request_to_validate = {'interface': 'admin', 'region_id': 1234, 'service_id': uuid.uuid4().hex, 'url': 'https://service.example.com:5000/'}
        self.assertRaises(exception.SchemaValidationError, self.update_endpoint_validator.validate, request_to_validate)
        request_to_validate = {'interface': 'admin', 'region': 1234, 'service_id': uuid.uuid4().hex, 'url': 'https://service.example.com:5000/'}
        self.assertRaises(exception.SchemaValidationError, self.update_endpoint_validator.validate, request_to_validate)