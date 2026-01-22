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
class UserValidationTestCase(unit.BaseTestCase):
    """Test for V3 User API validation."""

    def setUp(self):
        super(UserValidationTestCase, self).setUp()
        self.user_name = uuid.uuid4().hex
        create = identity_schema.user_create
        update = identity_schema.user_update
        self.create_user_validator = validators.SchemaValidator(create)
        self.update_user_validator = validators.SchemaValidator(update)

    def test_validate_user_create_request_succeeds(self):
        """Test that validating a user create request succeeds."""
        request_to_validate = {'name': self.user_name}
        self.create_user_validator.validate(request_to_validate)

    def test_validate_user_create_with_all_valid_parameters_succeeds(self):
        """Test that validating a user create request succeeds."""
        request_to_validate = unit.new_user_ref(domain_id=uuid.uuid4().hex, name=self.user_name)
        self.create_user_validator.validate(request_to_validate)

    def test_validate_user_create_fails_without_name(self):
        """Exception raised when validating a user without name."""
        request_to_validate = {'email': uuid.uuid4().hex}
        self.assertRaises(exception.SchemaValidationError, self.create_user_validator.validate, request_to_validate)

    def test_validate_user_create_succeeds_with_valid_enabled_formats(self):
        """Validate acceptable enabled formats in create user requests."""
        for enabled in _VALID_ENABLED_FORMATS:
            request_to_validate = {'name': self.user_name, 'enabled': enabled}
            self.create_user_validator.validate(request_to_validate)

    def test_validate_user_create_fails_with_invalid_enabled_formats(self):
        """Exception raised when enabled is not an acceptable format."""
        for invalid_enabled in _INVALID_ENABLED_FORMATS:
            request_to_validate = {'name': self.user_name, 'enabled': invalid_enabled}
            self.assertRaises(exception.SchemaValidationError, self.create_user_validator.validate, request_to_validate)

    def test_validate_user_create_succeeds_with_extra_attributes(self):
        """Validate extra parameters on user create requests."""
        request_to_validate = {'name': self.user_name, 'other_attr': uuid.uuid4().hex}
        self.create_user_validator.validate(request_to_validate)

    def test_validate_user_create_succeeds_with_password_of_zero_length(self):
        """Validate empty password on user create requests."""
        request_to_validate = {'name': self.user_name, 'password': ''}
        self.create_user_validator.validate(request_to_validate)

    def test_validate_user_create_succeeds_with_null_password(self):
        """Validate that password is nullable on create user."""
        request_to_validate = {'name': self.user_name, 'password': None}
        self.create_user_validator.validate(request_to_validate)

    def test_validate_user_create_fails_with_invalid_password_type(self):
        """Exception raised when user password is of the wrong type."""
        request_to_validate = {'name': self.user_name, 'password': True}
        self.assertRaises(exception.SchemaValidationError, self.create_user_validator.validate, request_to_validate)

    def test_validate_user_create_succeeds_with_null_description(self):
        """Validate that description can be nullable on create user."""
        request_to_validate = {'name': self.user_name, 'description': None}
        self.create_user_validator.validate(request_to_validate)

    def test_validate_user_create_fails_with_invalid_name(self):
        """Exception when validating a create request with invalid `name`."""
        for invalid_name in _INVALID_NAMES:
            request_to_validate = {'name': invalid_name}
            self.assertRaises(exception.SchemaValidationError, self.create_user_validator.validate, request_to_validate)

    def test_validate_user_update_succeeds(self):
        """Validate an update user request."""
        request_to_validate = {'email': uuid.uuid4().hex}
        self.update_user_validator.validate(request_to_validate)

    def test_validate_user_update_fails_with_no_parameters(self):
        """Exception raised when updating nothing."""
        request_to_validate = {}
        self.assertRaises(exception.SchemaValidationError, self.update_user_validator.validate, request_to_validate)

    def test_validate_user_update_succeeds_with_extra_parameters(self):
        """Validate user update requests with extra parameters."""
        request_to_validate = {'other_attr': uuid.uuid4().hex}
        self.update_user_validator.validate(request_to_validate)

    def test_validate_user_update_fails_with_invalid_name(self):
        """Exception when validating an update request with invalid `name`."""
        for invalid_name in _INVALID_NAMES:
            request_to_validate = {'name': invalid_name}
            self.assertRaises(exception.SchemaValidationError, self.update_user_validator.validate, request_to_validate)

    def test_user_create_succeeds_with_empty_options(self):
        request_to_validate = {'name': self.user_name, 'options': {}}
        self.create_user_validator.validate(request_to_validate)

    def test_user_create_options_fails_invalid_option(self):
        request_to_validate = {'name': self.user_name, 'options': {'whatever': True}}
        self.assertRaises(exception.SchemaValidationError, self.create_user_validator.validate, request_to_validate)

    def test_user_create_with_options_change_password_required(self):
        request_to_validate = {'name': self.user_name, 'options': {ro.IGNORE_CHANGE_PASSWORD_OPT.option_name: True}}
        self.create_user_validator.validate(request_to_validate)

    def test_user_create_options_change_password_required_wrong_type(self):
        request_to_validate = {'name': self.user_name, 'options': {ro.IGNORE_CHANGE_PASSWORD_OPT.option_name: 'whatever'}}
        self.assertRaises(exception.SchemaValidationError, self.create_user_validator.validate, request_to_validate)

    def test_user_create_options_change_password_required_none(self):
        request_to_validate = {'name': self.user_name, 'options': {ro.IGNORE_CHANGE_PASSWORD_OPT.option_name: None}}
        self.create_user_validator.validate(request_to_validate)

    def test_user_update_with_options_change_password_required(self):
        request_to_validate = {'options': {ro.IGNORE_CHANGE_PASSWORD_OPT.option_name: False}}
        self.update_user_validator.validate(request_to_validate)

    def test_user_create_with_options_lockout_password(self):
        request_to_validate = {'name': self.user_name, 'options': {ro.IGNORE_LOCKOUT_ATTEMPT_OPT.option_name: True}}
        self.create_user_validator.validate(request_to_validate)

    def test_user_update_with_options_lockout_password(self):
        request_to_validate = {'options': {ro.IGNORE_LOCKOUT_ATTEMPT_OPT.option_name: False}}
        self.update_user_validator.validate(request_to_validate)

    def test_user_update_with_two_options(self):
        request_to_validate = {'options': {ro.IGNORE_CHANGE_PASSWORD_OPT.option_name: True, ro.IGNORE_LOCKOUT_ATTEMPT_OPT.option_name: True}}
        self.update_user_validator.validate(request_to_validate)

    def test_user_create_with_two_options(self):
        request_to_validate = {'name': self.user_name, 'options': {ro.IGNORE_CHANGE_PASSWORD_OPT.option_name: False, ro.IGNORE_LOCKOUT_ATTEMPT_OPT.option_name: True}}
        self.create_user_validator.validate(request_to_validate)

    def test_user_create_with_mfa_rules(self):
        request_to_validate = {'name': self.user_name, 'options': {ro.MFA_RULES_OPT.option_name: [[uuid.uuid4().hex, uuid.uuid4().hex], [uuid.uuid4().hex]]}}
        self.create_user_validator.validate(request_to_validate)

    def test_user_update_with_mfa_rules(self):
        request_to_validate = {'options': {ro.MFA_RULES_OPT.option_name: [[uuid.uuid4().hex, uuid.uuid4().hex], [uuid.uuid4().hex]]}}
        self.update_user_validator.validate(request_to_validate)

    def test_user_create_with_mfa_rules_enabled(self):
        request_to_validate = {'name': self.user_name, 'options': {ro.MFA_ENABLED_OPT.option_name: True}}
        self.create_user_validator.validate(request_to_validate)

    def test_user_update_mfa_rules_enabled(self):
        request_to_validate = {'options': {ro.MFA_ENABLED_OPT.option_name: False}}
        self.update_user_validator.validate(request_to_validate)

    def test_user_option_validation_with_invalid_mfa_rules_fails(self):
        test_cases = [(True, TypeError), ([True, False], TypeError), ([[True], [True, False]], TypeError), ([['duplicate_array'] for x in range(0, 2)], ValueError), ([[uuid.uuid4().hex], []], ValueError), ([['duplicate' for x in range(0, 2)]], ValueError)]
        for ruleset, exception_class in test_cases:
            request_to_validate = {'options': {ro.MFA_RULES_OPT.option_name: ruleset}}
            self.assertRaises(exception.SchemaValidationError, self.update_user_validator.validate, request_to_validate)
            self.assertRaises(exception_class, ro._mfa_rules_validator_list_of_lists_of_strings_no_duplicates, ruleset)