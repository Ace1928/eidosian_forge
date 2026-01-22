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
class OAuth1ValidationTestCase(unit.BaseTestCase):
    """Test for V3 Identity OAuth1 API validation."""

    def setUp(self):
        super(OAuth1ValidationTestCase, self).setUp()
        create = oauth1_schema.consumer_create
        update = oauth1_schema.consumer_update
        authorize = oauth1_schema.request_token_authorize
        self.create_consumer_validator = validators.SchemaValidator(create)
        self.update_consumer_validator = validators.SchemaValidator(update)
        self.authorize_request_token_validator = validators.SchemaValidator(authorize)

    def test_validate_consumer_request_succeeds(self):
        """Test that we validate a consumer request successfully."""
        request_to_validate = {'description': uuid.uuid4().hex, 'name': uuid.uuid4().hex}
        self.create_consumer_validator.validate(request_to_validate)
        self.update_consumer_validator.validate(request_to_validate)

    def test_validate_consumer_request_with_no_parameters(self):
        """Test that schema validation with empty request body."""
        request_to_validate = {}
        self.create_consumer_validator.validate(request_to_validate)
        self.assertRaises(exception.SchemaValidationError, self.update_consumer_validator.validate, request_to_validate)

    def test_validate_consumer_request_with_invalid_description_fails(self):
        """Exception is raised when `description` as a non-string value."""
        for invalid_desc in _INVALID_DESC_FORMATS:
            request_to_validate = {'description': invalid_desc}
            self.assertRaises(exception.SchemaValidationError, self.create_consumer_validator.validate, request_to_validate)
            self.assertRaises(exception.SchemaValidationError, self.update_consumer_validator.validate, request_to_validate)

    def test_validate_update_consumer_request_fails_with_secret(self):
        """Exception raised when secret is given."""
        request_to_validate = {'secret': uuid.uuid4().hex}
        self.assertRaises(exception.SchemaValidationError, self.update_consumer_validator.validate, request_to_validate)

    def test_validate_consumer_request_with_none_desc(self):
        """Test that schema validation with None desc."""
        request_to_validate = {'description': None}
        self.create_consumer_validator.validate(request_to_validate)
        self.update_consumer_validator.validate(request_to_validate)

    def test_validate_authorize_request_token(self):
        request_to_validate = [{'id': '711aa6371a6343a9a43e8a310fbe4a6f'}, {'name': 'test_role'}]
        self.authorize_request_token_validator.validate(request_to_validate)

    def test_validate_authorize_request_token_with_additional_properties(self):
        request_to_validate = [{'id': '711aa6371a6343a9a43e8a310fbe4a6f', 'fake_key': 'fake_value'}]
        self.assertRaises(exception.SchemaValidationError, self.authorize_request_token_validator.validate, request_to_validate)

    def test_validate_authorize_request_token_with_id_and_name(self):
        request_to_validate = [{'id': '711aa6371a6343a9a43e8a310fbe4a6f', 'name': 'admin'}]
        self.assertRaises(exception.SchemaValidationError, self.authorize_request_token_validator.validate, request_to_validate)

    def test_validate_authorize_request_token_with_non_id_or_name(self):
        request_to_validate = [{'fake_key': 'fake_value'}]
        self.assertRaises(exception.SchemaValidationError, self.authorize_request_token_validator.validate, request_to_validate)