from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_two_methods(self):
    post_data = {'identity': {'methods': ['password', 'mapped'], 'password': {'user': {'name': 'admin', 'domain': {'name': 'Default'}, 'password': 'devstacker'}}}}
    schema.validate_issue_token_auth(post_data)