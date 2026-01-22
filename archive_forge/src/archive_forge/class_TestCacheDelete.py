from unittest.mock import call
from openstack import exceptions as sdk_exceptions
from osc_lib import exceptions
from openstackclient.image.v2 import cache
from openstackclient.tests.unit.image.v2 import fakes
class TestCacheDelete(fakes.TestImagev2):

    def setUp(self):
        super().setUp()
        self.image_client.cache_delete_image.return_value = None
        self.cmd = cache.DeleteCachedImage(self.app, None)

    def test_cache_delete(self):
        images = fakes.create_images(count=1)
        arglist = [images[0].id]
        verifylist = [('images', [images[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.image_client.find_image.side_effect = images
        self.cmd.take_action(parsed_args)
        self.image_client.find_image.assert_called_once_with(images[0].id, ignore_missing=False)
        self.image_client.cache_delete_image.assert_called_once_with(images[0].id)

    def test_cache_delete_multiple_images(self):
        images = fakes.create_images(count=3)
        arglist = [i.id for i in images]
        verifylist = [('images', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.image_client.find_image.side_effect = images
        self.cmd.take_action(parsed_args)
        calls = [call(i.id) for i in images]
        self.image_client.cache_delete_image.assert_has_calls(calls)

    def test_cache_delete_multiple_images_exception(self):
        images = fakes.create_images(count=2)
        arglist = [images[0].id, images[1].id, 'x-y-x']
        verifylist = [('images', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ret_find = [images[0], images[1], sdk_exceptions.ResourceNotFound()]
        self.image_client.find_image.side_effect = ret_find
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        calls = [call(i.id) for i in images]
        self.image_client.cache_delete_image.assert_has_calls(calls)