from keystoneclient import base
def _create_common_request(self, service_provider, token_id):
    headers = {'Content-Type': 'application/json'}
    body = {'auth': {'identity': {'methods': ['token'], 'token': {'id': token_id}}, 'scope': {'service_provider': {'id': base.getid(service_provider)}}}}
    return (headers, body)