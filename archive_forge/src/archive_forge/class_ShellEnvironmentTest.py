import json
import tempfile
from unittest import mock
import io
from oslo_serialization import base64
import testtools
from testtools import matchers
from urllib import error
import yaml
from heatclient.common import template_utils
from heatclient.common import utils
from heatclient import exc
class ShellEnvironmentTest(testtools.TestCase):
    template_a = b'{"heat_template_version": "2013-05-23"}'

    def collect_links(self, env, content, url, env_base_url=''):
        jenv = yaml.safe_load(env)
        files = {}
        if url:

            def side_effect(args):
                if url == args:
                    return io.BytesIO(content)
            with mock.patch('urllib.request.urlopen') as mock_url:
                mock_url.side_effect = side_effect
                template_utils.resolve_environment_urls(jenv.get('resource_registry'), files, env_base_url)
                self.assertEqual(content.decode('utf-8'), files[url])
        else:
            template_utils.resolve_environment_urls(jenv.get('resource_registry'), files, env_base_url)

    @mock.patch('urllib.request.urlopen')
    def test_ignore_env_keys(self, mock_url):
        env_file = '/home/my/dir/env.yaml'
        env = b'\n        resource_registry:\n          resources:\n            bar:\n              hooks: pre_create\n              restricted_actions: replace\n        '
        mock_url.return_value = io.BytesIO(env)
        _, env_dict = template_utils.process_environment_and_files(env_file)
        self.assertEqual({'resource_registry': {'resources': {'bar': {'hooks': 'pre_create', 'restricted_actions': 'replace'}}}}, env_dict)
        mock_url.assert_called_with('file://%s' % env_file)

    @mock.patch('urllib.request.urlopen')
    def test_process_environment_file(self, mock_url):
        env_file = '/home/my/dir/env.yaml'
        env = b'\n        resource_registry:\n          "OS::Thingy": "file:///home/b/a.yaml"\n        '
        mock_url.side_effect = [io.BytesIO(env), io.BytesIO(self.template_a), io.BytesIO(self.template_a)]
        files, env_dict = template_utils.process_environment_and_files(env_file)
        self.assertEqual({'resource_registry': {'OS::Thingy': 'file:///home/b/a.yaml'}}, env_dict)
        self.assertEqual(self.template_a.decode('utf-8'), files['file:///home/b/a.yaml'])
        mock_url.assert_has_calls([mock.call('file://%s' % env_file), mock.call('file:///home/b/a.yaml'), mock.call('file:///home/b/a.yaml')])

    @mock.patch('urllib.request.urlopen')
    def test_process_environment_relative_file(self, mock_url):
        env_file = '/home/my/dir/env.yaml'
        env_url = 'file:///home/my/dir/env.yaml'
        env = b'\n        resource_registry:\n          "OS::Thingy": a.yaml\n        '
        mock_url.side_effect = [io.BytesIO(env), io.BytesIO(self.template_a), io.BytesIO(self.template_a)]
        self.assertEqual(env_url, utils.normalise_file_path_to_url(env_file))
        self.assertEqual('file:///home/my/dir', utils.base_url_for_url(env_url))
        files, env_dict = template_utils.process_environment_and_files(env_file)
        self.assertEqual({'resource_registry': {'OS::Thingy': 'file:///home/my/dir/a.yaml'}}, env_dict)
        self.assertEqual(self.template_a.decode('utf-8'), files['file:///home/my/dir/a.yaml'])
        mock_url.assert_has_calls([mock.call(env_url), mock.call('file:///home/my/dir/a.yaml'), mock.call('file:///home/my/dir/a.yaml')])

    def test_process_multiple_environment_files_container(self):
        env_list_tracker = []
        env_paths = ['/home/my/dir/env.yaml']
        files, env = template_utils.process_multiple_environments_and_files(env_paths, env_list_tracker=env_list_tracker, fetch_env_files=False)
        self.assertEqual(env_paths, env_list_tracker)
        self.assertEqual({}, files)
        self.assertEqual({}, env)

    @mock.patch('urllib.request.urlopen')
    def test_process_environment_relative_file_up(self, mock_url):
        env_file = '/home/my/dir/env.yaml'
        env_url = 'file:///home/my/dir/env.yaml'
        env = b'\n        resource_registry:\n          "OS::Thingy": ../bar/a.yaml\n        '
        mock_url.side_effect = [io.BytesIO(env), io.BytesIO(self.template_a), io.BytesIO(self.template_a)]
        env_url = 'file://%s' % env_file
        self.assertEqual(env_url, utils.normalise_file_path_to_url(env_file))
        self.assertEqual('file:///home/my/dir', utils.base_url_for_url(env_url))
        files, env_dict = template_utils.process_environment_and_files(env_file)
        self.assertEqual({'resource_registry': {'OS::Thingy': 'file:///home/my/bar/a.yaml'}}, env_dict)
        self.assertEqual(self.template_a.decode('utf-8'), files['file:///home/my/bar/a.yaml'])
        mock_url.assert_has_calls([mock.call(env_url), mock.call('file:///home/my/bar/a.yaml'), mock.call('file:///home/my/bar/a.yaml')])

    @mock.patch('urllib.request.urlopen')
    def test_process_environment_url(self, mock_url):
        env = b'\n        resource_registry:\n            "OS::Thingy": "a.yaml"\n        '
        url = 'http://no.where/some/path/to/file.yaml'
        tmpl_url = 'http://no.where/some/path/to/a.yaml'
        mock_url.side_effect = [io.BytesIO(env), io.BytesIO(self.template_a), io.BytesIO(self.template_a)]
        files, env_dict = template_utils.process_environment_and_files(url)
        self.assertEqual({'resource_registry': {'OS::Thingy': tmpl_url}}, env_dict)
        self.assertEqual(self.template_a.decode('utf-8'), files[tmpl_url])
        mock_url.assert_has_calls([mock.call(url), mock.call(tmpl_url), mock.call(tmpl_url)])

    @mock.patch('urllib.request.urlopen')
    def test_process_environment_empty_file(self, mock_url):
        env_file = '/home/my/dir/env.yaml'
        env = b''
        mock_url.return_value = io.BytesIO(env)
        files, env_dict = template_utils.process_environment_and_files(env_file)
        self.assertEqual({}, env_dict)
        self.assertEqual({}, files)
        mock_url.assert_called_with('file://%s' % env_file)

    def test_no_process_environment_and_files(self):
        files, env = template_utils.process_environment_and_files()
        self.assertEqual({}, env)
        self.assertEqual({}, files)

    @mock.patch('urllib.request.urlopen')
    def test_process_multiple_environments_and_files(self, mock_url):
        env_file1 = '/home/my/dir/env1.yaml'
        env_file2 = '/home/my/dir/env2.yaml'
        env1 = b'\n        parameters:\n          "param1": "value1"\n        resource_registry:\n          "OS::Thingy1": "file:///home/b/a.yaml"\n        '
        env2 = b'\n        parameters:\n          "param2": "value2"\n        resource_registry:\n          "OS::Thingy2": "file:///home/b/b.yaml"\n        '
        mock_url.side_effect = [io.BytesIO(env1), io.BytesIO(self.template_a), io.BytesIO(self.template_a), io.BytesIO(env2), io.BytesIO(self.template_a), io.BytesIO(self.template_a)]
        files, env = template_utils.process_multiple_environments_and_files([env_file1, env_file2])
        self.assertEqual({'resource_registry': {'OS::Thingy1': 'file:///home/b/a.yaml', 'OS::Thingy2': 'file:///home/b/b.yaml'}, 'parameters': {'param1': 'value1', 'param2': 'value2'}}, env)
        self.assertEqual(self.template_a.decode('utf-8'), files['file:///home/b/a.yaml'])
        self.assertEqual(self.template_a.decode('utf-8'), files['file:///home/b/b.yaml'])
        mock_url.assert_has_calls([mock.call('file://%s' % env_file1), mock.call('file:///home/b/a.yaml'), mock.call('file:///home/b/a.yaml'), mock.call('file://%s' % env_file2), mock.call('file:///home/b/b.yaml'), mock.call('file:///home/b/b.yaml')])

    @mock.patch('urllib.request.urlopen')
    def test_process_multiple_environments_default_resources(self, mock_url):
        env_file1 = '/home/my/dir/env1.yaml'
        env_file2 = '/home/my/dir/env2.yaml'
        env1 = b'\n        resource_registry:\n          resources:\n            resource1:\n              "OS::Thingy1": "file:///home/b/a.yaml"\n            resource2:\n              "OS::Thingy2": "file:///home/b/b.yaml"\n        '
        env2 = b'\n        resource_registry:\n          resources:\n            resource1:\n              "OS::Thingy3": "file:///home/b/a.yaml"\n            resource2:\n              "OS::Thingy4": "file:///home/b/b.yaml"\n        '
        mock_url.side_effect = [io.BytesIO(env1), io.BytesIO(self.template_a), io.BytesIO(self.template_a), io.BytesIO(self.template_a), io.BytesIO(self.template_a), io.BytesIO(env2), io.BytesIO(self.template_a), io.BytesIO(self.template_a), io.BytesIO(self.template_a), io.BytesIO(self.template_a)]
        files, env = template_utils.process_multiple_environments_and_files([env_file1, env_file2])
        self.assertEqual({'resource_registry': {'resources': {'resource1': {'OS::Thingy1': 'file:///home/b/a.yaml', 'OS::Thingy3': 'file:///home/b/a.yaml'}, 'resource2': {'OS::Thingy2': 'file:///home/b/b.yaml', 'OS::Thingy4': 'file:///home/b/b.yaml'}}}}, env)
        self.assertEqual(self.template_a.decode('utf-8'), files['file:///home/b/a.yaml'])
        self.assertEqual(self.template_a.decode('utf-8'), files['file:///home/b/b.yaml'])
        mock_url.assert_has_calls([mock.call('file://%s' % env_file1), mock.call('file:///home/b/a.yaml'), mock.call('file:///home/b/b.yaml'), mock.call('file:///home/b/a.yaml'), mock.call('file:///home/b/b.yaml'), mock.call('file://%s' % env_file2), mock.call('file:///home/b/a.yaml'), mock.call('file:///home/b/b.yaml'), mock.call('file:///home/b/a.yaml'), mock.call('file:///home/b/b.yaml')], any_order=True)

    def test_no_process_multiple_environments_and_files(self):
        files, env = template_utils.process_multiple_environments_and_files()
        self.assertEqual({}, env)
        self.assertEqual({}, files)

    def test_process_multiple_environments_and_files_from_object(self):
        env_object = 'http://no.where/path/to/env.yaml'
        env1 = b'\n        parameters:\n          "param1": "value1"\n        resource_registry:\n          "OS::Thingy1": "b/a.yaml"\n        '
        self.object_requested = False

        def env_path_is_object(object_url):
            return True

        def object_request(method, object_url):
            self.object_requested = True
            self.assertEqual('GET', method)
            self.assertTrue(object_url.startswith('http://no.where/path/to/'))
            if object_url == env_object:
                return env1
            else:
                return self.template_a
        files, env = template_utils.process_multiple_environments_and_files(env_paths=[env_object], env_path_is_object=env_path_is_object, object_request=object_request)
        self.assertEqual({'resource_registry': {'OS::Thingy1': 'http://no.where/path/to/b/a.yaml'}, 'parameters': {'param1': 'value1'}}, env)
        self.assertEqual(self.template_a.decode('utf-8'), files['http://no.where/path/to/b/a.yaml'])

    @mock.patch('urllib.request.urlopen')
    def test_process_multiple_environments_and_files_tracker(self, mock_url):
        env_file1 = '/home/my/dir/env1.yaml'
        env1 = b'\n        parameters:\n          "param1": "value1"\n        resource_registry:\n          "OS::Thingy1": "file:///home/b/a.yaml"\n        '
        mock_url.side_effect = [io.BytesIO(env1), io.BytesIO(self.template_a), io.BytesIO(self.template_a)]
        env_file_list = []
        files, env = template_utils.process_multiple_environments_and_files([env_file1], env_list_tracker=env_file_list)
        expected_env = {'parameters': {'param1': 'value1'}, 'resource_registry': {'OS::Thingy1': 'file:///home/b/a.yaml'}}
        self.assertEqual(expected_env, env)
        self.assertEqual(self.template_a.decode('utf-8'), files['file:///home/b/a.yaml'])
        self.assertEqual(['file:///home/my/dir/env1.yaml'], env_file_list)
        self.assertIn('file:///home/my/dir/env1.yaml', files)
        self.assertEqual(expected_env, json.loads(files['file:///home/my/dir/env1.yaml']))
        mock_url.assert_has_calls([mock.call('file://%s' % env_file1), mock.call('file:///home/b/a.yaml'), mock.call('file:///home/b/a.yaml')])

    @mock.patch('urllib.request.urlopen')
    def test_process_environment_relative_file_tracker(self, mock_url):
        env_file = '/home/my/dir/env.yaml'
        env_url = 'file:///home/my/dir/env.yaml'
        env = b'\n        resource_registry:\n          "OS::Thingy": a.yaml\n        '
        mock_url.side_effect = [io.BytesIO(env), io.BytesIO(self.template_a), io.BytesIO(self.template_a)]
        self.assertEqual(env_url, utils.normalise_file_path_to_url(env_file))
        self.assertEqual('file:///home/my/dir', utils.base_url_for_url(env_url))
        env_file_list = []
        files, env = template_utils.process_multiple_environments_and_files([env_file], env_list_tracker=env_file_list)
        expected_env = {'resource_registry': {'OS::Thingy': 'file:///home/my/dir/a.yaml'}}
        self.assertEqual(expected_env, env)
        self.assertEqual(self.template_a.decode('utf-8'), files['file:///home/my/dir/a.yaml'])
        self.assertEqual(['file:///home/my/dir/env.yaml'], env_file_list)
        self.assertEqual(json.dumps(expected_env), files['file:///home/my/dir/env.yaml'])
        mock_url.assert_has_calls([mock.call(env_url), mock.call('file:///home/my/dir/a.yaml'), mock.call('file:///home/my/dir/a.yaml')])

    @mock.patch('urllib.request.urlopen')
    def test_process_multiple_environments_empty_registry(self, mock_url):
        env_file1 = '/home/my/dir/env1.yaml'
        env_file2 = '/home/my/dir/env2.yaml'
        env1 = b'\n        resource_registry:\n          "OS::Thingy1": "file:///home/b/a.yaml"\n        '
        env2 = b'\n        resource_registry:\n        '
        mock_url.side_effect = [io.BytesIO(env1), io.BytesIO(self.template_a), io.BytesIO(self.template_a), io.BytesIO(env2)]
        env_file_list = []
        files, env = template_utils.process_multiple_environments_and_files([env_file1, env_file2], env_list_tracker=env_file_list)
        expected_env = {'resource_registry': {'OS::Thingy1': 'file:///home/b/a.yaml'}}
        self.assertEqual(expected_env, env)
        self.assertEqual(self.template_a.decode('utf-8'), files['file:///home/b/a.yaml'])
        self.assertEqual(['file:///home/my/dir/env1.yaml', 'file:///home/my/dir/env2.yaml'], env_file_list)
        self.assertIn('file:///home/my/dir/env1.yaml', files)
        self.assertIn('file:///home/my/dir/env2.yaml', files)
        self.assertEqual(expected_env, json.loads(files['file:///home/my/dir/env1.yaml']))
        mock_url.assert_has_calls([mock.call('file://%s' % env_file1), mock.call('file:///home/b/a.yaml'), mock.call('file:///home/b/a.yaml'), mock.call('file://%s' % env_file2)])

    def test_global_files(self):
        url = 'file:///home/b/a.yaml'
        env = '\n        resource_registry:\n          "OS::Thingy": "%s"\n        ' % url
        self.collect_links(env, self.template_a, url)

    def test_nested_files(self):
        url = 'file:///home/b/a.yaml'
        env = '\n        resource_registry:\n          resources:\n            freddy:\n              "OS::Thingy": "%s"\n        ' % url
        self.collect_links(env, self.template_a, url)

    def test_http_url(self):
        url = 'http://no.where/container/a.yaml'
        env = '\n        resource_registry:\n          "OS::Thingy": "%s"\n        ' % url
        self.collect_links(env, self.template_a, url)

    def test_with_base_url(self):
        url = 'ftp://no.where/container/a.yaml'
        env = '\n        resource_registry:\n          base_url: "ftp://no.where/container/"\n          resources:\n            server_for_me:\n              "OS::Thingy": a.yaml\n        '
        self.collect_links(env, self.template_a, url)

    def test_with_built_in_provider(self):
        env = '\n        resource_registry:\n          resources:\n            server_for_me:\n              "OS::Thingy": OS::Compute::Server\n        '
        self.collect_links(env, self.template_a, None)

    def test_with_env_file_base_url_file(self):
        url = 'file:///tmp/foo/a.yaml'
        env = '\n        resource_registry:\n          resources:\n            server_for_me:\n              "OS::Thingy": a.yaml\n        '
        env_base_url = 'file:///tmp/foo'
        self.collect_links(env, self.template_a, url, env_base_url)

    def test_with_env_file_base_url_http(self):
        url = 'http://no.where/path/to/a.yaml'
        env = '\n        resource_registry:\n          resources:\n            server_for_me:\n              "OS::Thingy": to/a.yaml\n        '
        env_base_url = 'http://no.where/path'
        self.collect_links(env, self.template_a, url, env_base_url)

    def test_unsupported_protocol(self):
        env = '\n        resource_registry:\n          "OS::Thingy": "sftp://no.where/dev/null/a.yaml"\n        '
        jenv = yaml.safe_load(env)
        fields = {'files': {}}
        self.assertRaises(exc.CommandError, template_utils.get_file_contents, jenv['resource_registry'], fields)