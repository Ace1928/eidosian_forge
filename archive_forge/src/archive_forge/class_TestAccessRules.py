from keystonemiddleware.auth_token import _path_matches
from keystonemiddleware.tests.unit import utils
class TestAccessRules(utils.BaseTestCase):

    def test_path_matches(self):
        good_matches = [('/v2/servers', '/v2/servers'), ('/v2/servers/123', '/v2/servers/{server_id}'), ('/v2/servers/123/', '/v2/servers/{server_id}/'), ('/v2/servers/123', '/v2/servers/*'), ('/v2/servers/123/', '/v2/servers/*/'), ('/v2/servers/123', '/v2/servers/**'), ('/v2/servers/123/', '/v2/servers/**'), ('/v2/servers/123/456', '/v2/servers/**'), ('/v2/servers', '**'), ('/v2/servers/', '**'), ('/v2/servers/123', '**'), ('/v2/servers/123/456', '**'), ('/v2/servers/123/volume/456', '**'), ('/v2/servers/123/456', '/v2/*/*/*'), ('/v2/123/servers/466', '/v2/{project_id}/servers/{server_id}')]
        for request, pattern in good_matches:
            self.assertIsNotNone(_path_matches(request, pattern))
        bad_matches = [('/v2/servers/someuuid', '/v2/servers'), ('/v2/servers//', '/v2/servers/{server_id}'), ('/v2/servers/123/', '/v2/servers/{server_id}'), ('/v2/servers/123/456', '/v2/servers/{server_id}'), ('/v2/servers/123/456', '/v2/servers/*'), ('/v2/servers', 'v2/servers'), ('/v2/servers/123/456/789', '/v2/*/*/*'), ('/v2/servers/123/', '/v2/*/*/*'), ('/v2/servers/', '/v2/servers/{server_id}'), ('/v2/servers', '/v2/servers/{server_id}')]
        for request, pattern in bad_matches:
            self.assertIsNone(_path_matches(request, pattern))