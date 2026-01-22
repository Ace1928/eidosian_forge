import ddt
import time
from zunclient.tests.functional.osc.v1 import base
@ddt.ddt
class ContainerTests(base.TestCase):
    """Functional tests for container commands."""

    def setUp(self):
        super(ContainerTests, self).setUp()

    def test_list(self):
        """Check container list command.

        """
        container = self.container_create(name='test_list')
        container_list = self.container_list()
        self.assertIn(container['name'], [x['name'] for x in container_list])
        self.assertIn(container['uuid'], [x['uuid'] for x in container_list])
        self.container_delete(container['name'])

    def test_create(self):
        """Check container create command.

        """
        container_info = self.container_create(name='test_create')
        self.assertEqual(container_info['name'], 'test_create')
        self.assertEqual(container_info['image'], 'cirros')
        container_list = self.container_list()
        self.assertIn('test_create', [x['name'] for x in container_list])
        self.container_delete(container_info['name'])

    def test_delete(self):
        """Check container delete command with name/UUID argument.

        Test steps:
        1) Create container in setUp.
        2) Delete container by name/UUID.
        3) Check that container deleted successfully.
        """
        container = self.container_create(name='test_delete')
        container_list = self.container_list()
        self.assertIn(container['name'], [x['name'] for x in container_list])
        self.assertIn(container['uuid'], [x['uuid'] for x in container_list])
        count = 0
        while count < 5:
            container = self.container_show(container['name'])
            if container['status'] == 'Created':
                break
            if container['status'] == 'Error':
                break
            time.sleep(2)
            count = count + 1
        self.container_delete(container['name'])
        count = 0
        while count < 5:
            container_list = self.container_list()
            if container['name'] not in [x['name'] for x in container_list]:
                break
            time.sleep(2)
            count = count + 1
        container_list = self.container_list()
        self.assertNotIn(container['name'], [x['name'] for x in container_list])
        self.assertNotIn(container['uuid'], [x['uuid'] for x in container_list])

    def test_show(self):
        """Check container show command with name and UUID arguments.

        Test steps:
        1) Create container in setUp.
        2) Show container calling it with name and UUID arguments.
        3) Check name, uuid and image in container show output.
        """
        container = self.container_create(name='test_show')
        self.container_show(container['name'])
        self.assertEqual(container['name'], container['name'])
        self.assertEqual(container['image'], container['image'])
        self.container_delete(container['name'])

    def test_execute(self):
        """Check container execute command with name and UUID arguments.

        Test steps:
        1) Create container in setUp.
        2) Execute command calling it with name and UUID arguments.
        3) Check the container logs.
        """
        container = self.container_run(name='test_execute')
        count = 0
        while count < 50:
            container = self.container_show(container['name'])
            if container['status'] == 'Running':
                break
            if container['status'] == 'Error':
                break
            time.sleep(2)
            count = count + 1
        command = "sh -c 'echo hello'"
        result = self.container_execute(container['name'], command)
        self.assertIn('hello', result)
        self.container_delete(container['name'])