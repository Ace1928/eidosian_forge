import argparse
import io
import json
import os
from unittest import mock
import subprocess
import tempfile
import testtools
from glanceclient import exc
from glanceclient import shell
import glanceclient.v1.client as client
import glanceclient.v1.images
import glanceclient.v1.shell as v1shell
from glanceclient.tests import utils
class ShellStdinHandlingTests(testtools.TestCase):

    def _fake_update_func(self, *args, **kwargs):
        """Replace glanceclient.images.update with a fake.

        To determine the parameters that would be supplied with the update
        request.
        """
        self.collected_args = (args, kwargs)
        return args[0]

    def setUp(self):
        super(ShellStdinHandlingTests, self).setUp()
        self.api = utils.FakeAPI(fixtures)
        self.gc = client.Client('http://fakeaddress.com')
        self.gc.images = glanceclient.v1.images.ImageManager(self.api)
        self.real_sys_stdin_fd = os.dup(0)
        dev_null = open('/dev/null')
        self.dev_null_fd = dev_null.fileno()
        os.dup2(dev_null.fileno(), 0)
        self.real_update_func = self.gc.images.update
        self.collected_args = []
        self.gc.images.update = self._fake_update_func

    def tearDown(self):
        """Restore stdin and gc.images.update to their pretest states."""
        super(ShellStdinHandlingTests, self).tearDown()

        def try_close(fd):
            try:
                os.close(fd)
            except OSError:
                pass
        os.dup2(self.real_sys_stdin_fd, 0)
        try_close(self.real_sys_stdin_fd)
        try_close(self.dev_null_fd)
        self.gc.images.update = self.real_update_func

    def _do_update(self, image='96d2c7e1-de4e-4612-8aa2-ba26610c804e'):
        """call v1/shell's do_image_update function."""
        v1shell.do_image_update(self.gc, argparse.Namespace(image=image, name='testimagerename', property={}, purge_props=False, human_readable=False, file=None, progress=False))

    def test_image_delete_deleted(self):
        self.assertRaises(exc.CommandError, v1shell.do_image_delete, self.gc, argparse.Namespace(images=['70aa106f-3750-4d7c-a5ce-0a535ac08d0a']))

    def test_image_update_closed_stdin(self):
        """Test image update with a closed stdin.

        Supply glanceclient with a closed stdin, and perform an image
        update to an active image. Glanceclient should not attempt to read
        stdin.
        """
        os.close(0)
        self._do_update()
        self.assertTrue('data' not in self.collected_args[1] or self.collected_args[1]['data'] is None)

    def test_image_update_opened_stdin(self):
        """Test image update with an opened stdin.

        Supply glanceclient with a stdin, and perform an image
        update to an active image. Glanceclient should not allow it.
        """
        self.assertRaises(SystemExit, v1shell.do_image_update, self.gc, argparse.Namespace(image='96d2c7e1-de4e-4612-8aa2-ba26610c804e', property={}))

    def test_image_update_data_is_read_from_file(self):
        """Ensure that data is read from a file."""
        try:
            f = open(tempfile.mktemp(), 'w+')
            f.write('Some Data')
            f.flush()
            f.seek(0)
            os.dup2(f.fileno(), 0)
            self._do_update('44d2c7e1-de4e-4612-8aa2-ba26610c444f')
            self.assertIn('data', self.collected_args[1])
            self.assertIsInstance(self.collected_args[1]['data'], io.IOBase)
            self.assertEqual(b'Some Data', self.collected_args[1]['data'].read())
        finally:
            try:
                f.close()
                os.remove(f.name)
            except Exception:
                pass

    def test_image_update_data_is_read_from_pipe(self):
        """Ensure that data is read from a pipe."""
        try:
            process = subprocess.Popen(['/bin/echo', 'Some Data'], stdout=subprocess.PIPE)
            os.dup2(process.stdout.fileno(), 0)
            self._do_update('44d2c7e1-de4e-4612-8aa2-ba26610c444f')
            self.assertIn('data', self.collected_args[1])
            self.assertIsInstance(self.collected_args[1]['data'], io.IOBase)
            self.assertEqual(b'Some Data\n', self.collected_args[1]['data'].read())
        finally:
            try:
                process.stdout.close()
            except OSError:
                pass