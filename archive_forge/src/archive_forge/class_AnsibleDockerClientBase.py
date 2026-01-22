from __future__ import (absolute_import, division, print_function)
import abc
import json
import shlex
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._api.auth import resolve_repository_name
from ansible_collections.community.docker.plugins.module_utils.util import (  # noqa: F401, pylint: disable=unused-import
class AnsibleDockerClientBase(object):

    def __init__(self, common_args, min_docker_api_version=None):
        self._environment = {}
        if common_args['tls_hostname']:
            self._environment['DOCKER_TLS_HOSTNAME'] = common_args['tls_hostname']
        if common_args['api_version'] and common_args['api_version'] != 'auto':
            self._environment['DOCKER_API_VERSION'] = common_args['api_version']
        self._cli = common_args.get('docker_cli')
        if self._cli is None:
            try:
                self._cli = get_bin_path('docker')
            except ValueError:
                self.fail('Cannot find docker CLI in path. Please provide it explicitly with the docker_cli parameter')
        self._cli_base = [self._cli]
        self._cli_base.extend(['--host', common_args['docker_host']])
        if common_args['validate_certs']:
            self._cli_base.append('--tlsverify')
        elif common_args['tls']:
            self._cli_base.append('--tls')
        if common_args['ca_path']:
            self._cli_base.extend(['--tlscacert', common_args['ca_path']])
        if common_args['client_cert']:
            self._cli_base.extend(['--tlscert', common_args['client_cert']])
        if common_args['client_key']:
            self._cli_base.extend(['--tlskey', common_args['client_key']])
        if common_args['cli_context']:
            self._cli_base.extend(['--context', common_args['cli_context']])
        dummy, self._version, dummy = self.call_cli_json('version', '--format', '{{ json . }}', check_rc=True)
        self._info = None
        self.docker_api_version_str = self._version['Server']['ApiVersion']
        self.docker_api_version = LooseVersion(self.docker_api_version_str)
        min_docker_api_version = min_docker_api_version or '1.25'
        if self.docker_api_version < LooseVersion(min_docker_api_version):
            self.fail('Docker API version is %s. Minimum version required is %s.' % (self.docker_api_version_str, min_docker_api_version))

    def log(self, msg, pretty_print=False):
        pass

    def get_cli(self):
        return self._cli

    def get_version_info(self):
        return self._version

    def _compose_cmd(self, args):
        return self._cli_base + list(args)

    def _compose_cmd_str(self, args):
        return ' '.join((shlex.quote(a) for a in self._compose_cmd(args)))

    @abc.abstractmethod
    def call_cli(self, *args, **kwargs):
        pass

    def call_cli_json(self, *args, **kwargs):
        warn_on_stderr = kwargs.pop('warn_on_stderr', False)
        rc, stdout, stderr = self.call_cli(*args, **kwargs)
        if warn_on_stderr and stderr:
            self.warn(to_native(stderr))
        try:
            data = json.loads(stdout)
        except Exception as exc:
            self.fail('Error while parsing JSON output of {cmd}: {exc}\nJSON output: {stdout}'.format(cmd=self._compose_cmd_str(args), exc=to_native(exc), stdout=to_native(stdout)))
        return (rc, data, stderr)

    def call_cli_json_stream(self, *args, **kwargs):
        warn_on_stderr = kwargs.pop('warn_on_stderr', False)
        rc, stdout, stderr = self.call_cli(*args, **kwargs)
        if warn_on_stderr and stderr:
            self.warn(to_native(stderr))
        result = []
        try:
            for line in stdout.splitlines():
                line = line.strip()
                if line.startswith(b'{'):
                    result.append(json.loads(line))
        except Exception as exc:
            self.fail('Error while parsing JSON output of {cmd}: {exc}\nJSON output: {stdout}'.format(cmd=self._compose_cmd_str(args), exc=to_native(exc), stdout=to_native(stdout)))
        return (rc, result, stderr)

    @abc.abstractmethod
    def fail(self, msg, **kwargs):
        pass

    @abc.abstractmethod
    def warn(self, msg):
        pass

    @abc.abstractmethod
    def deprecate(self, msg, version=None, date=None, collection_name=None):
        pass

    def get_cli_info(self):
        if self._info is None:
            dummy, self._info, dummy = self.call_cli_json('info', '--format', '{{ json . }}', check_rc=True)
        return self._info

    def get_client_plugin_info(self, component):
        for plugin in self.get_cli_info()['ClientInfo'].get('Plugins') or []:
            if plugin.get('Name') == component:
                return plugin
        return None

    def _image_lookup(self, name, tag):
        """
        Including a tag in the name parameter sent to the Docker SDK for Python images method
        does not work consistently. Instead, get the result set for name and manually check
        if the tag exists.
        """
        dummy, images, dummy = self.call_cli_json_stream('image', 'ls', '--format', '{{ json . }}', '--no-trunc', '--filter', 'reference={0}'.format(name), check_rc=True)
        if tag:
            lookup = '%s:%s' % (name, tag)
            lookup_digest = '%s@%s' % (name, tag)
            response = images
            images = []
            for image in response:
                if image.get('Tag') == tag or image.get('Digest') == tag:
                    images = [image]
                    break
        return images

    def find_image(self, name, tag):
        """
        Lookup an image (by name and tag) and return the inspection results.
        """
        if not name:
            return None
        self.log('Find image %s:%s' % (name, tag))
        images = self._image_lookup(name, tag)
        if not images:
            registry, repo_name = resolve_repository_name(name)
            if registry == 'docker.io':
                self.log('Check for docker.io image: %s' % repo_name)
                images = self._image_lookup(repo_name, tag)
                if not images and repo_name.startswith('library/'):
                    lookup = repo_name[len('library/'):]
                    self.log('Check for docker.io image: %s' % lookup)
                    images = self._image_lookup(lookup, tag)
                if not images:
                    lookup = '%s/%s' % (registry, repo_name)
                    self.log('Check for docker.io image: %s' % lookup)
                    images = self._image_lookup(lookup, tag)
                if not images and '/' not in repo_name:
                    lookup = '%s/library/%s' % (registry, repo_name)
                    self.log('Check for docker.io image: %s' % lookup)
                    images = self._image_lookup(lookup, tag)
        if len(images) > 1:
            self.fail('Daemon returned more than one result for %s:%s' % (name, tag))
        if len(images) == 1:
            rc, image, stderr = self.call_cli_json('image', 'inspect', images[0]['ID'])
            if not image:
                self.log('Image %s:%s not found.' % (name, tag))
                return None
            if rc != 0:
                self.fail('Error inspecting image %s:%s - %s' % (name, tag, to_native(stderr)))
            return image[0]
        self.log('Image %s:%s not found.' % (name, tag))
        return None

    def find_image_by_id(self, image_id, accept_missing_image=False):
        """
        Lookup an image (by ID) and return the inspection results.
        """
        if not image_id:
            return None
        self.log('Find image %s (by ID)' % image_id)
        rc, image, stderr = self.call_cli_json('image', 'inspect', image_id)
        if not image:
            if not accept_missing_image:
                self.fail('Error inspecting image ID %s - %s' % (image_id, to_native(stderr)))
            self.log('Image %s not found.' % image_id)
            return None
        if rc != 0:
            self.fail('Error inspecting image ID %s - %s' % (image_id, to_native(stderr)))
        return image[0]