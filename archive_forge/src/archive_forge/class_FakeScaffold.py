from pecan.tests import PecanTestCase
class FakeScaffold(object):

    def copy_to(self, project_name):
        assert project_name == 'default'