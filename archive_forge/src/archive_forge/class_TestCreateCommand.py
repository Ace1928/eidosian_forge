from pecan.tests import PecanTestCase
class TestCreateCommand(PecanTestCase):

    def test_run(self):
        from pecan.commands import CreateCommand

        class FakeArg(object):
            project_name = 'default'
            template_name = 'default'

        class FakeScaffold(object):

            def copy_to(self, project_name):
                assert project_name == 'default'

        class FakeManager(object):
            scaffolds = {'default': FakeScaffold}
        c = CreateCommand()
        c.manager = FakeManager()
        c.run(FakeArg())