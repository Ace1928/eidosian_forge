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
class ChangePasswordValidationTestCase(unit.BaseTestCase):
    """Test for Change Password API validation."""

    def setUp(self):
        super(ChangePasswordValidationTestCase, self).setUp()
        self.original_password = uuid.uuid4().hex
        self.password = uuid.uuid4().hex
        change = identity_schema.password_change
        self.change_password_validator = validators.SchemaValidator(change)

    def test_validate_password_change_request_succeeds(self):
        """Test that validating a password change request succeeds."""
        request_to_validate = {'original_password': self.original_password, 'password': self.password}
        self.change_password_validator.validate(request_to_validate)

    def test_validate_password_change_fails_without_all_fields(self):
        """Test that validating a password change fails without all values."""
        request_to_validate = {'original_password': self.original_password}
        self.assertRaises(exception.SchemaValidationError, self.change_password_validator.validate, request_to_validate)
        request_to_validate = {'password': self.password}
        self.assertRaises(exception.SchemaValidationError, self.change_password_validator.validate, request_to_validate)

    def test_validate_password_change_fails_with_invalid_values(self):
        """Test that validating a password change fails with bad values."""
        request_to_validate = {'original_password': None, 'password': None}
        self.assertRaises(exception.SchemaValidationError, self.change_password_validator.validate, request_to_validate)
        request_to_validate = {'original_password': 42, 'password': True}
        self.assertRaises(exception.SchemaValidationError, self.change_password_validator.validate, request_to_validate)