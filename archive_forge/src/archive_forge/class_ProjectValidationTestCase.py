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
class ProjectValidationTestCase(unit.BaseTestCase):
    """Test for V3 Project API validation."""

    def setUp(self):
        super(ProjectValidationTestCase, self).setUp()
        self.project_name = 'My Project'
        create = resource_schema.project_create
        update = resource_schema.project_update
        self.create_project_validator = validators.SchemaValidator(create)
        self.update_project_validator = validators.SchemaValidator(update)

    def test_validate_project_request(self):
        """Test that we validate a project with `name` in request."""
        request_to_validate = {'name': self.project_name}
        self.create_project_validator.validate(request_to_validate)

    def test_validate_project_request_without_name_fails(self):
        """Validate project request fails without name."""
        request_to_validate = {'enabled': True}
        self.assertRaises(exception.SchemaValidationError, self.create_project_validator.validate, request_to_validate)

    def test_validate_project_request_with_enabled(self):
        """Validate `enabled` as boolean-like values for projects."""
        for valid_enabled in _VALID_ENABLED_FORMATS:
            request_to_validate = {'name': self.project_name, 'enabled': valid_enabled}
            self.create_project_validator.validate(request_to_validate)

    def test_validate_project_request_with_invalid_enabled_fails(self):
        """Exception is raised when `enabled` isn't a boolean-like value."""
        for invalid_enabled in _INVALID_ENABLED_FORMATS:
            request_to_validate = {'name': self.project_name, 'enabled': invalid_enabled}
            self.assertRaises(exception.SchemaValidationError, self.create_project_validator.validate, request_to_validate)

    def test_validate_project_request_with_valid_description(self):
        """Test that we validate `description` in create project requests."""
        request_to_validate = {'name': self.project_name, 'description': 'My Project'}
        self.create_project_validator.validate(request_to_validate)

    def test_validate_project_request_with_invalid_description_fails(self):
        """Exception is raised when `description` as a non-string value."""
        request_to_validate = {'name': self.project_name, 'description': False}
        self.assertRaises(exception.SchemaValidationError, self.create_project_validator.validate, request_to_validate)

    def test_validate_project_request_with_name_too_long(self):
        """Exception is raised when `name` is too long."""
        long_project_name = 'a' * 65
        request_to_validate = {'name': long_project_name}
        self.assertRaises(exception.SchemaValidationError, self.create_project_validator.validate, request_to_validate)

    def test_validate_project_create_fails_with_invalid_name(self):
        """Exception when validating a create request with invalid `name`."""
        for invalid_name in _INVALID_NAMES + ['a' * 65]:
            request_to_validate = {'name': invalid_name}
            self.assertRaises(exception.SchemaValidationError, self.create_project_validator.validate, request_to_validate)

    def test_validate_project_create_with_tags(self):
        request_to_validate = {'name': uuid.uuid4().hex, 'tags': ['foo', 'bar']}
        self.create_project_validator.validate(request_to_validate)

    def test_validate_project_create_with_tags_invalid_char(self):
        invalid_chars = [',', '/', ',foo', 'foo/bar']
        for char in invalid_chars:
            tag = uuid.uuid4().hex + char
            request_to_validate = {'name': uuid.uuid4().hex, 'tags': ['foo', tag]}
            self.assertRaises(exception.SchemaValidationError, self.create_project_validator.validate, request_to_validate)

    def test_validate_project_create_with_tag_name_too_long(self):
        invalid_name = 'a' * 256
        request_to_validate = {'name': uuid.uuid4().hex, 'tags': ['foo', invalid_name]}
        self.assertRaises(exception.SchemaValidationError, self.create_project_validator.validate, request_to_validate)

    def test_validate_project_create_with_too_many_tags(self):
        tags = [uuid.uuid4().hex for _ in range(81)]
        request_to_validate = {'name': uuid.uuid4().hex, 'tags': tags}
        self.assertRaises(exception.SchemaValidationError, self.create_project_validator.validate, request_to_validate)

    def test_validate_project_request_with_valid_parent_id(self):
        """Test that we validate `parent_id` in create project requests."""
        request_to_validate = {'name': self.project_name, 'parent_id': None}
        self.create_project_validator.validate(request_to_validate)
        request_to_validate = {'name': self.project_name, 'parent_id': uuid.uuid4().hex}
        self.create_project_validator.validate(request_to_validate)

    def test_validate_project_request_with_invalid_parent_id_fails(self):
        """Exception is raised when `parent_id` as a non-id value."""
        request_to_validate = {'name': self.project_name, 'parent_id': False}
        self.assertRaises(exception.SchemaValidationError, self.create_project_validator.validate, request_to_validate)
        request_to_validate = {'name': self.project_name, 'parent_id': 'fake project'}
        self.assertRaises(exception.SchemaValidationError, self.create_project_validator.validate, request_to_validate)

    def test_validate_project_update_request(self):
        """Test that we validate a project update request."""
        request_to_validate = {'domain_id': uuid.uuid4().hex}
        self.update_project_validator.validate(request_to_validate)

    def test_validate_project_update_request_with_no_parameters_fails(self):
        """Exception is raised when updating project without parameters."""
        request_to_validate = {}
        self.assertRaises(exception.SchemaValidationError, self.update_project_validator.validate, request_to_validate)

    def test_validate_project_update_request_with_name_too_long_fails(self):
        """Exception raised when updating a project with `name` too long."""
        long_project_name = 'a' * 65
        request_to_validate = {'name': long_project_name}
        self.assertRaises(exception.SchemaValidationError, self.update_project_validator.validate, request_to_validate)

    def test_validate_project_update_fails_with_invalid_name(self):
        """Exception when validating an update request with invalid `name`."""
        for invalid_name in _INVALID_NAMES:
            request_to_validate = {'name': invalid_name}
            self.assertRaises(exception.SchemaValidationError, self.update_project_validator.validate, request_to_validate)

    def test_validate_project_update_with_tags(self):
        request_to_validate = {'name': uuid.uuid4().hex, 'tags': ['foo', 'bar']}
        self.update_project_validator.validate(request_to_validate)

    def test_validate_project_update_with_tags_invalid_char(self):
        invalid_chars = [',', '/']
        for char in invalid_chars:
            tag = uuid.uuid4().hex + char
            request_to_validate = {'name': uuid.uuid4().hex, 'tags': ['foo', tag]}
            self.assertRaises(exception.SchemaValidationError, self.update_project_validator.validate, request_to_validate)

    def test_validate_project_update_with_tag_name_too_long(self):
        invalid_name = 'a' * 256
        request_to_validate = {'name': uuid.uuid4().hex, 'tags': ['foo', invalid_name]}
        self.assertRaises(exception.SchemaValidationError, self.update_project_validator.validate, request_to_validate)

    def test_validate_project_update_with_too_many_tags(self):
        tags = [uuid.uuid4().hex for _ in range(81)]
        request_to_validate = {'name': uuid.uuid4().hex, 'tags': tags}
        self.assertRaises(exception.SchemaValidationError, self.update_project_validator.validate, request_to_validate)

    def test_validate_project_create_request_with_valid_domain_id(self):
        """Test that we validate `domain_id` in create project requests."""
        for domain_id in [None, uuid.uuid4().hex]:
            request_to_validate = {'name': self.project_name, 'domain_id': domain_id}
            self.create_project_validator.validate(request_to_validate)

    def test_validate_project_request_with_invalid_domain_id_fails(self):
        """Exception is raised when `domain_id` is a non-id value."""
        for domain_id in [False, 'fake_project']:
            request_to_validate = {'name': self.project_name, 'domain_id': domain_id}
            self.assertRaises(exception.SchemaValidationError, self.create_project_validator.validate, request_to_validate)