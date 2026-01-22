from breezy import repository
from breezy.bzr.remote import RemoteRepositoryFormat
from breezy.tests import default_transport, multiply_tests, test_server
from breezy.tests.per_controldir.test_controldir import TestCaseWithControlDir
from breezy.transport import memory
def all_repository_format_scenarios():
    """Return a list of test scenarios for parameterising repository tests.
    """
    all_formats = repository.format_registry._get_all()
    format_scenarios = formats_to_scenarios([('', format) for format in all_formats], default_transport, None)
    format_scenarios.extend(formats_to_scenarios([('-default', RemoteRepositoryFormat())], test_server.SmartTCPServer_for_testing, test_server.ReadonlySmartTCPServer_for_testing, memory.MemoryServer))
    format_scenarios.extend(formats_to_scenarios([('-v2', RemoteRepositoryFormat())], test_server.SmartTCPServer_for_testing_v2_only, test_server.ReadonlySmartTCPServer_for_testing_v2_only, memory.MemoryServer))
    return format_scenarios