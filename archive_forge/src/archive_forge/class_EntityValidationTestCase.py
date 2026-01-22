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
class EntityValidationTestCase(unit.BaseTestCase):

    def setUp(self):
        super(EntityValidationTestCase, self).setUp()
        self.resource_name = 'some resource name'
        self.description = 'Some valid description'
        self.valid_enabled = True
        self.valid_url = 'http://example.com'
        self.valid_email = 'joe@example.com'
        self.create_schema_validator = validators.SchemaValidator(entity_create)
        self.update_schema_validator = validators.SchemaValidator(entity_update)

    def test_create_entity_with_all_valid_parameters_validates(self):
        """Validate all parameter values against test schema."""
        request_to_validate = {'name': self.resource_name, 'description': self.description, 'enabled': self.valid_enabled, 'url': self.valid_url, 'email': self.valid_email}
        self.create_schema_validator.validate(request_to_validate)

    def test_create_entity_with_only_required_valid_parameters_validates(self):
        """Validate correct for only parameters values against test schema."""
        request_to_validate = {'name': self.resource_name}
        self.create_schema_validator.validate(request_to_validate)

    def test_create_entity_with_name_too_long_raises_exception(self):
        """Validate long names.

        Validate that an exception is raised when validating a string of 255+
        characters passed in as a name.
        """
        invalid_name = 'a' * 256
        request_to_validate = {'name': invalid_name}
        self.assertRaises(exception.SchemaValidationError, self.create_schema_validator.validate, request_to_validate)

    def test_create_entity_with_name_too_short_raises_exception(self):
        """Validate short names.

        Test that an exception is raised when passing a string of length
        zero as a name parameter.
        """
        request_to_validate = {'name': ''}
        self.assertRaises(exception.SchemaValidationError, self.create_schema_validator.validate, request_to_validate)

    def test_create_entity_with_unicode_name_validates(self):
        """Test that we successfully validate a unicode string."""
        request_to_validate = {'name': u'αβγδ'}
        self.create_schema_validator.validate(request_to_validate)

    def test_create_entity_with_invalid_enabled_format_raises_exception(self):
        """Validate invalid enabled formats.

        Test that an exception is raised when passing invalid boolean-like
        values as `enabled`.
        """
        for format in _INVALID_ENABLED_FORMATS:
            request_to_validate = {'name': self.resource_name, 'enabled': format}
            self.assertRaises(exception.SchemaValidationError, self.create_schema_validator.validate, request_to_validate)

    def test_create_entity_with_valid_enabled_formats_validates(self):
        """Validate valid enabled formats.

        Test that we have successful validation on boolean values for
        `enabled`.
        """
        for valid_enabled in _VALID_ENABLED_FORMATS:
            request_to_validate = {'name': self.resource_name, 'enabled': valid_enabled}
            self.create_schema_validator.validate(request_to_validate)

    def test_create_entity_with_valid_urls_validates(self):
        """Test that proper urls are successfully validated."""
        for valid_url in _VALID_URLS:
            request_to_validate = {'name': self.resource_name, 'url': valid_url}
            self.create_schema_validator.validate(request_to_validate)

    def test_create_entity_with_invalid_urls_fails(self):
        """Test that an exception is raised when validating improper urls."""
        for invalid_url in _INVALID_URLS:
            request_to_validate = {'name': self.resource_name, 'url': invalid_url}
            self.assertRaises(exception.SchemaValidationError, self.create_schema_validator.validate, request_to_validate)

    def test_create_entity_with_valid_email_validates(self):
        """Validate email address.

        Test that we successfully validate properly formatted email
        addresses.
        """
        request_to_validate = {'name': self.resource_name, 'email': self.valid_email}
        self.create_schema_validator.validate(request_to_validate)

    def test_create_entity_with_invalid_email_fails(self):
        """Validate invalid email address.

        Test that an exception is raised when validating improperly
        formatted email addresses.
        """
        request_to_validate = {'name': self.resource_name, 'email': 'some invalid email value'}
        self.assertRaises(exception.SchemaValidationError, self.create_schema_validator.validate, request_to_validate)

    def test_create_entity_with_valid_id_strings(self):
        """Validate acceptable id strings."""
        valid_id_strings = [str(uuid.uuid4()), uuid.uuid4().hex, 'default']
        for valid_id in valid_id_strings:
            request_to_validate = {'name': self.resource_name, 'id_string': valid_id}
            self.create_schema_validator.validate(request_to_validate)

    def test_create_entity_with_invalid_id_strings(self):
        """Exception raised when using invalid id strings."""
        long_string = 'A' * 65
        invalid_id_strings = ['', long_string]
        for invalid_id in invalid_id_strings:
            request_to_validate = {'name': self.resource_name, 'id_string': invalid_id}
            self.assertRaises(exception.SchemaValidationError, self.create_schema_validator.validate, request_to_validate)

    def test_create_entity_with_null_id_string(self):
        """Validate that None is an acceptable optional string type."""
        request_to_validate = {'name': self.resource_name, 'id_string': None}
        self.create_schema_validator.validate(request_to_validate)

    def test_create_entity_with_null_string_succeeds(self):
        """Exception raised when passing None on required id strings."""
        request_to_validate = {'name': self.resource_name, 'id_string': None}
        self.create_schema_validator.validate(request_to_validate)

    def test_update_entity_with_no_parameters_fails(self):
        """At least one parameter needs to be present for an update."""
        request_to_validate = {}
        self.assertRaises(exception.SchemaValidationError, self.update_schema_validator.validate, request_to_validate)

    def test_update_entity_with_all_parameters_valid_validates(self):
        """Simulate updating an entity by ID."""
        request_to_validate = {'name': self.resource_name, 'description': self.description, 'enabled': self.valid_enabled, 'url': self.valid_url, 'email': self.valid_email}
        self.update_schema_validator.validate(request_to_validate)

    def test_update_entity_with_a_valid_required_parameter_validates(self):
        """Succeed if a valid required parameter is provided."""
        request_to_validate = {'name': self.resource_name}
        self.update_schema_validator.validate(request_to_validate)

    def test_update_entity_with_invalid_required_parameter_fails(self):
        """Fail if a provided required parameter is invalid."""
        request_to_validate = {'name': 'a' * 256}
        self.assertRaises(exception.SchemaValidationError, self.update_schema_validator.validate, request_to_validate)

    def test_update_entity_with_a_null_optional_parameter_validates(self):
        """Optional parameters can be null to removed the value."""
        request_to_validate = {'email': None}
        self.update_schema_validator.validate(request_to_validate)

    def test_update_entity_with_a_required_null_parameter_fails(self):
        """The `name` parameter can't be null."""
        request_to_validate = {'name': None}
        self.assertRaises(exception.SchemaValidationError, self.update_schema_validator.validate, request_to_validate)

    def test_update_entity_with_a_valid_optional_parameter_validates(self):
        """Succeed with only a single valid optional parameter."""
        request_to_validate = {'email': self.valid_email}
        self.update_schema_validator.validate(request_to_validate)

    def test_update_entity_with_invalid_optional_parameter_fails(self):
        """Fail when an optional parameter is invalid."""
        request_to_validate = {'email': 0}
        self.assertRaises(exception.SchemaValidationError, self.update_schema_validator.validate, request_to_validate)