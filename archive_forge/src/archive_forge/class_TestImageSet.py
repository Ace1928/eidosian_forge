import copy
import io
import tempfile
from unittest import mock
from cinderclient import api_versions
from openstack import exceptions as sdk_exceptions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.image.v2 import image as _image
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
class TestImageSet(TestImage):
    project = identity_fakes.FakeProject.create_one_project()
    domain = identity_fakes.FakeDomain.create_one_domain()
    _image = image_fakes.create_one_image({'tags': []})

    def setUp(self):
        super().setUp()
        self.project_mock.get.return_value = self.project
        self.domain_mock.get.return_value = self.domain
        self.image_client.find_image.return_value = self._image
        self.app.client_manager.auth_ref = mock.Mock(project_id=self.project.id)
        self.cmd = _image.SetImage(self.app, None)

    def test_image_set_no_options(self):
        arglist = ['0f41529e-7c12-4de8-be2d-181abb825b3c']
        verifylist = [('image', '0f41529e-7c12-4de8-be2d-181abb825b3c')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        self.image_client.update_image.assert_called_once_with(self._image.id)

    def test_image_set_membership_option_accept(self):
        membership = image_fakes.create_one_image_member(attrs={'image_id': '0f41529e-7c12-4de8-be2d-181abb825b3c', 'member_id': self.project.id})
        self.image_client.update_member.return_value = membership
        arglist = ['--accept', self._image.id]
        verifylist = [('membership', 'accepted'), ('image', self._image.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.image_client.update_member.assert_called_once_with(image=self._image.id, member=self.app.client_manager.auth_ref.project_id, status='accepted')
        self.image_client.update_image.assert_called_with(self._image.id)

    def test_image_set_membership_option_reject(self):
        membership = image_fakes.create_one_image_member(attrs={'image_id': '0f41529e-7c12-4de8-be2d-181abb825b3c', 'member_id': self.project.id})
        self.image_client.update_member.return_value = membership
        arglist = ['--reject', '0f41529e-7c12-4de8-be2d-181abb825b3c']
        verifylist = [('membership', 'rejected'), ('image', '0f41529e-7c12-4de8-be2d-181abb825b3c')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.image_client.update_member.assert_called_once_with(image=self._image.id, member=self.app.client_manager.auth_ref.project_id, status='rejected')
        self.image_client.update_image.assert_called_with(self._image.id)

    def test_image_set_membership_option_pending(self):
        membership = image_fakes.create_one_image_member(attrs={'image_id': '0f41529e-7c12-4de8-be2d-181abb825b3c', 'member_id': self.project.id})
        self.image_client.update_member.return_value = membership
        arglist = ['--pending', '0f41529e-7c12-4de8-be2d-181abb825b3c']
        verifylist = [('membership', 'pending'), ('image', '0f41529e-7c12-4de8-be2d-181abb825b3c')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.image_client.update_member.assert_called_once_with(image=self._image.id, member=self.app.client_manager.auth_ref.project_id, status='pending')
        self.image_client.update_image.assert_called_with(self._image.id)

    def test_image_set_options(self):
        arglist = ['--name', 'new-name', '--min-disk', '2', '--min-ram', '4', '--container-format', 'ovf', '--disk-format', 'vmdk', '--project', self.project.name, '--project-domain', self.domain.id, self._image.id]
        verifylist = [('name', 'new-name'), ('min_disk', 2), ('min_ram', 4), ('container_format', 'ovf'), ('disk_format', 'vmdk'), ('project', self.project.name), ('project_domain', self.domain.id), ('image', self._image.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'name': 'new-name', 'owner_id': self.project.id, 'min_disk': 2, 'min_ram': 4, 'container_format': 'ovf', 'disk_format': 'vmdk'}
        self.image_client.update_image.assert_called_with(self._image.id, **kwargs)
        self.assertIsNone(result)

    def test_image_set_with_unexist_project(self):
        self.project_mock.get.side_effect = exceptions.NotFound(None)
        self.project_mock.find.side_effect = exceptions.NotFound(None)
        arglist = ['--project', 'unexist_owner', '0f41529e-7c12-4de8-be2d-181abb825b3c']
        verifylist = [('project', 'unexist_owner'), ('image', '0f41529e-7c12-4de8-be2d-181abb825b3c')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_image_set_bools1(self):
        arglist = ['--protected', '--private', 'graven']
        verifylist = [('is_protected', True), ('visibility', 'private'), ('image', 'graven')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'is_protected': True, 'visibility': 'private'}
        self.image_client.update_image.assert_called_with(self._image.id, **kwargs)
        self.assertIsNone(result)

    def test_image_set_bools2(self):
        arglist = ['--unprotected', '--public', 'graven']
        verifylist = [('is_protected', False), ('visibility', 'public'), ('image', 'graven')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'is_protected': False, 'visibility': 'public'}
        self.image_client.update_image.assert_called_with(self._image.id, **kwargs)
        self.assertIsNone(result)

    def test_image_set_properties(self):
        arglist = ['--property', 'Alpha=1', '--property', 'Beta=2', 'graven']
        verifylist = [('properties', {'Alpha': '1', 'Beta': '2'}), ('image', 'graven')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'Alpha': '1', 'Beta': '2'}
        self.image_client.update_image.assert_called_with(self._image.id, **kwargs)
        self.assertIsNone(result)

    def test_image_set_fake_properties(self):
        arglist = ['--architecture', 'z80', '--instance-id', '12345', '--kernel-id', '67890', '--os-distro', 'cpm', '--os-version', '2.2H', '--ramdisk-id', 'xyzpdq', 'graven']
        verifylist = [('architecture', 'z80'), ('instance_id', '12345'), ('kernel_id', '67890'), ('os_distro', 'cpm'), ('os_version', '2.2H'), ('ramdisk_id', 'xyzpdq'), ('image', 'graven')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'architecture': 'z80', 'instance_id': '12345', 'kernel_id': '67890', 'os_distro': 'cpm', 'os_version': '2.2H', 'ramdisk_id': 'xyzpdq'}
        self.image_client.update_image.assert_called_with(self._image.id, **kwargs)
        self.assertIsNone(result)

    def test_image_set_tag(self):
        arglist = ['--tag', 'test-tag', 'graven']
        verifylist = [('tags', ['test-tag']), ('image', 'graven')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'tags': ['test-tag']}
        self.image_client.update_image.assert_called_with(self._image.id, **kwargs)
        self.assertIsNone(result)

    def test_image_set_activate(self):
        arglist = ['--tag', 'test-tag', '--activate', 'graven']
        verifylist = [('tags', ['test-tag']), ('image', 'graven')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'tags': ['test-tag']}
        self.image_client.reactivate_image.assert_called_with(self._image.id)
        self.image_client.update_image.assert_called_with(self._image.id, **kwargs)
        self.assertIsNone(result)

    def test_image_set_deactivate(self):
        arglist = ['--tag', 'test-tag', '--deactivate', 'graven']
        verifylist = [('tags', ['test-tag']), ('image', 'graven')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'tags': ['test-tag']}
        self.image_client.deactivate_image.assert_called_with(self._image.id)
        self.image_client.update_image.assert_called_with(self._image.id, **kwargs)
        self.assertIsNone(result)

    def test_image_set_tag_merge(self):
        old_image = self._image
        old_image['tags'] = ['old1', 'new2']
        self.image_client.find_image.return_value = old_image
        arglist = ['--tag', 'test-tag', 'graven']
        verifylist = [('tags', ['test-tag']), ('image', 'graven')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'tags': ['old1', 'new2', 'test-tag']}
        a, k = self.image_client.update_image.call_args
        self.assertEqual(self._image.id, a[0])
        self.assertIn('tags', k)
        self.assertEqual(set(kwargs['tags']), set(k['tags']))
        self.assertIsNone(result)

    def test_image_set_tag_merge_dupe(self):
        old_image = self._image
        old_image['tags'] = ['old1', 'new2']
        self.image_client.find_image.return_value = old_image
        arglist = ['--tag', 'old1', 'graven']
        verifylist = [('tags', ['old1']), ('image', 'graven')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'tags': ['new2', 'old1']}
        a, k = self.image_client.update_image.call_args
        self.assertEqual(self._image.id, a[0])
        self.assertIn('tags', k)
        self.assertEqual(set(kwargs['tags']), set(k['tags']))
        self.assertIsNone(result)

    def test_image_set_dead_options(self):
        arglist = ['--visibility', '1-mile', 'graven']
        verifylist = [('dead_visibility', '1-mile'), ('image', 'graven')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_image_set_numeric_options_to_zero(self):
        arglist = ['--min-disk', '0', '--min-ram', '0', 'graven']
        verifylist = [('min_disk', 0), ('min_ram', 0), ('image', 'graven')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'min_disk': 0, 'min_ram': 0}
        self.image_client.update_image.assert_called_with(self._image.id, **kwargs)
        self.assertIsNone(result)

    def test_image_set_hidden(self):
        arglist = ['--hidden', '--public', 'graven']
        verifylist = [('is_hidden', True), ('visibility', 'public'), ('image', 'graven')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'is_hidden': True, 'visibility': 'public'}
        self.image_client.update_image.assert_called_with(self._image.id, **kwargs)
        self.assertIsNone(result)

    def test_image_set_unhidden(self):
        arglist = ['--unhidden', '--public', 'graven']
        verifylist = [('is_hidden', False), ('visibility', 'public'), ('image', 'graven')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'is_hidden': False, 'visibility': 'public'}
        self.image_client.update_image.assert_called_with(self._image.id, **kwargs)
        self.assertIsNone(result)