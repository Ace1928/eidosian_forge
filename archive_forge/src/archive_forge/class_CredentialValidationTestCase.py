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
class CredentialValidationTestCase(unit.BaseTestCase):
    """Test for V3 Credential API validation."""

    def setUp(self):
        super(CredentialValidationTestCase, self).setUp()
        create = credential_schema.credential_create
        update = credential_schema.credential_update
        self.create_credential_validator = validators.SchemaValidator(create)
        self.update_credential_validator = validators.SchemaValidator(update)

    def test_validate_credential_succeeds(self):
        """Test that we validate a credential request."""
        request_to_validate = {'blob': 'some string', 'project_id': uuid.uuid4().hex, 'type': 'ec2', 'user_id': uuid.uuid4().hex}
        self.create_credential_validator.validate(request_to_validate)

    def test_validate_credential_without_blob_fails(self):
        """Exception raised without `blob` in create request."""
        request_to_validate = {'type': 'ec2', 'user_id': uuid.uuid4().hex}
        self.assertRaises(exception.SchemaValidationError, self.create_credential_validator.validate, request_to_validate)

    def test_validate_credential_without_user_id_fails(self):
        """Exception raised without `user_id` in create request."""
        request_to_validate = {'blob': 'some credential blob', 'type': 'ec2'}
        self.assertRaises(exception.SchemaValidationError, self.create_credential_validator.validate, request_to_validate)

    def test_validate_credential_without_type_fails(self):
        """Exception raised without `type` in create request."""
        request_to_validate = {'blob': 'some credential blob', 'user_id': uuid.uuid4().hex}
        self.assertRaises(exception.SchemaValidationError, self.create_credential_validator.validate, request_to_validate)

    def test_validate_credential_ec2_without_project_id_fails(self):
        """Validate `project_id` is required for ec2.

        Test that a SchemaValidationError is raised when type is ec2
        and no `project_id` is provided in create request.
        """
        request_to_validate = {'blob': 'some credential blob', 'type': 'ec2', 'user_id': uuid.uuid4().hex}
        self.assertRaises(exception.SchemaValidationError, self.create_credential_validator.validate, request_to_validate)

    def test_validate_credential_with_project_id_succeeds(self):
        """Test that credential request works for all types."""
        cred_types = ['ec2', 'cert', uuid.uuid4().hex]
        for c_type in cred_types:
            request_to_validate = {'blob': 'some blob', 'project_id': uuid.uuid4().hex, 'type': c_type, 'user_id': uuid.uuid4().hex}
            self.create_credential_validator.validate(request_to_validate)

    def test_validate_credential_non_ec2_without_project_id_succeeds(self):
        """Validate `project_id` is not required for non-ec2.

        Test that create request without `project_id` succeeds for any
        non-ec2 credential.
        """
        cred_types = ['cert', uuid.uuid4().hex]
        for c_type in cred_types:
            request_to_validate = {'blob': 'some blob', 'type': c_type, 'user_id': uuid.uuid4().hex}
            self.create_credential_validator.validate(request_to_validate)

    def test_validate_credential_with_extra_parameters_succeeds(self):
        """Validate create request with extra parameters."""
        request_to_validate = {'blob': 'some string', 'extra': False, 'project_id': uuid.uuid4().hex, 'type': 'ec2', 'user_id': uuid.uuid4().hex}
        self.create_credential_validator.validate(request_to_validate)

    def test_validate_credential_update_succeeds(self):
        """Test that a credential request is properly validated."""
        request_to_validate = {'blob': 'some string', 'project_id': uuid.uuid4().hex, 'type': 'ec2', 'user_id': uuid.uuid4().hex}
        self.update_credential_validator.validate(request_to_validate)

    def test_validate_credential_update_without_parameters_fails(self):
        """Exception is raised on update without parameters."""
        request_to_validate = {}
        self.assertRaises(exception.SchemaValidationError, self.update_credential_validator.validate, request_to_validate)

    def test_validate_credential_update_with_extra_parameters_succeeds(self):
        """Validate credential update with extra parameters."""
        request_to_validate = {'blob': 'some string', 'extra': False, 'project_id': uuid.uuid4().hex, 'type': 'ec2', 'user_id': uuid.uuid4().hex}
        self.update_credential_validator.validate(request_to_validate)