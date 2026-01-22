from zaqarclient.transport import api
class FakeApi(api.Api):
    label = 'v1'
    schema = {'test_operation': {'ref': 'test/{name}', 'method': 'GET', 'properties': {'name': {'type': 'string'}, 'address': {'type': 'string'}}, 'additionalProperties': False, 'required': ['name']}}