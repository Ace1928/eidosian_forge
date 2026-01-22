import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import credentials
from google.auth import downscoped
from google.auth import exceptions
from google.auth import transport
class TestCredentialAccessBoundary(object):

    def test_constructor(self):
        availability_condition = make_availability_condition(EXPRESSION, TITLE, DESCRIPTION)
        access_boundary_rule = make_access_boundary_rule(AVAILABLE_RESOURCE, AVAILABLE_PERMISSIONS, availability_condition)
        rules = [access_boundary_rule]
        credential_access_boundary = make_credential_access_boundary(rules)
        assert credential_access_boundary.rules == tuple(rules)

    def test_setters(self):
        availability_condition = make_availability_condition(EXPRESSION, TITLE, DESCRIPTION)
        access_boundary_rule = make_access_boundary_rule(AVAILABLE_RESOURCE, AVAILABLE_PERMISSIONS, availability_condition)
        rules = [access_boundary_rule]
        other_availability_condition = make_availability_condition(OTHER_EXPRESSION, OTHER_TITLE, OTHER_DESCRIPTION)
        other_access_boundary_rule = make_access_boundary_rule(OTHER_AVAILABLE_RESOURCE, OTHER_AVAILABLE_PERMISSIONS, other_availability_condition)
        other_rules = [other_access_boundary_rule]
        credential_access_boundary = make_credential_access_boundary(rules)
        credential_access_boundary.rules = other_rules
        assert credential_access_boundary.rules == tuple(other_rules)

    def test_add_rule(self):
        availability_condition = make_availability_condition(EXPRESSION, TITLE, DESCRIPTION)
        access_boundary_rule = make_access_boundary_rule(AVAILABLE_RESOURCE, AVAILABLE_PERMISSIONS, availability_condition)
        rules = [access_boundary_rule] * 9
        credential_access_boundary = make_credential_access_boundary(rules)
        additional_access_boundary_rule = make_access_boundary_rule(OTHER_AVAILABLE_RESOURCE, OTHER_AVAILABLE_PERMISSIONS)
        credential_access_boundary.add_rule(additional_access_boundary_rule)
        assert len(credential_access_boundary.rules) == 10
        assert credential_access_boundary.rules[9] == additional_access_boundary_rule

    def test_add_rule_invalid_value(self):
        availability_condition = make_availability_condition(EXPRESSION, TITLE, DESCRIPTION)
        access_boundary_rule = make_access_boundary_rule(AVAILABLE_RESOURCE, AVAILABLE_PERMISSIONS, availability_condition)
        rules = [access_boundary_rule] * 10
        credential_access_boundary = make_credential_access_boundary(rules)
        with pytest.raises(ValueError) as excinfo:
            credential_access_boundary.add_rule(access_boundary_rule)
        assert excinfo.match('Credential access boundary rules can have a maximum of 10 rules.')
        assert len(credential_access_boundary.rules) == 10

    def test_add_rule_invalid_type(self):
        availability_condition = make_availability_condition(EXPRESSION, TITLE, DESCRIPTION)
        access_boundary_rule = make_access_boundary_rule(AVAILABLE_RESOURCE, AVAILABLE_PERMISSIONS, availability_condition)
        rules = [access_boundary_rule]
        credential_access_boundary = make_credential_access_boundary(rules)
        with pytest.raises(TypeError) as excinfo:
            credential_access_boundary.add_rule('invalid')
        assert excinfo.match("The provided rule does not contain a valid 'google.auth.downscoped.AccessBoundaryRule'.")
        assert len(credential_access_boundary.rules) == 1
        assert credential_access_boundary.rules[0] == access_boundary_rule

    def test_invalid_rules_type(self):
        with pytest.raises(TypeError) as excinfo:
            make_credential_access_boundary(['invalid'])
        assert excinfo.match("List of rules provided do not contain a valid 'google.auth.downscoped.AccessBoundaryRule'.")

    def test_invalid_rules_value(self):
        availability_condition = make_availability_condition(EXPRESSION, TITLE, DESCRIPTION)
        access_boundary_rule = make_access_boundary_rule(AVAILABLE_RESOURCE, AVAILABLE_PERMISSIONS, availability_condition)
        too_many_rules = [access_boundary_rule] * 11
        with pytest.raises(ValueError) as excinfo:
            make_credential_access_boundary(too_many_rules)
        assert excinfo.match('Credential access boundary rules can have a maximum of 10 rules.')

    def test_to_json(self):
        availability_condition = make_availability_condition(EXPRESSION, TITLE, DESCRIPTION)
        access_boundary_rule = make_access_boundary_rule(AVAILABLE_RESOURCE, AVAILABLE_PERMISSIONS, availability_condition)
        rules = [access_boundary_rule]
        credential_access_boundary = make_credential_access_boundary(rules)
        assert credential_access_boundary.to_json() == {'accessBoundary': {'accessBoundaryRules': [{'availablePermissions': AVAILABLE_PERMISSIONS, 'availableResource': AVAILABLE_RESOURCE, 'availabilityCondition': {'expression': EXPRESSION, 'title': TITLE, 'description': DESCRIPTION}}]}}