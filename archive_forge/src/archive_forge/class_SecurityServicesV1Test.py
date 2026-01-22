from manilaclient.tests.unit import utils
class SecurityServicesV1Test(utils.TestCase):

    def test_import_v1_security_services_module(self):
        try:
            from manilaclient.v1 import security_services
        except Exception as e:
            msg = "module 'manilaclient.v1.security_services' cannot be imported with error: %s" % str(e)
            assert False, msg
        for cls in ('SecurityService', 'SecurityServiceManager'):
            msg = "Module 'security_services' has no '%s' attr." % cls
            self.assertTrue(hasattr(security_services, cls), msg)