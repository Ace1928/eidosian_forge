from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
from ansible_collections.community.docker.plugins.module_utils._api.auth import (
class ImagePusher(DockerBaseClass):

    def __init__(self, client):
        super(ImagePusher, self).__init__()
        self.client = client
        self.check_mode = self.client.check_mode
        parameters = self.client.module.params
        self.name = parameters['name']
        self.tag = parameters['tag']
        if is_image_name_id(self.name):
            self.client.fail('Cannot push an image by ID')
        if not is_valid_tag(self.tag, allow_empty=True):
            self.client.fail('"{0}" is not a valid docker tag!'.format(self.tag))
        repo, repo_tag = parse_repository_tag(self.name)
        if repo_tag:
            self.name = repo
            self.tag = repo_tag
        if is_image_name_id(self.tag):
            self.client.fail('Cannot push an image by digest')
        if not is_valid_tag(self.tag, allow_empty=False):
            self.client.fail('"{0}" is not a valid docker tag!'.format(self.tag))

    def push(self):
        image = self.client.find_image(name=self.name, tag=self.tag)
        if not image:
            self.client.fail('Cannot find image %s:%s' % (self.name, self.tag))
        results = dict(changed=False, actions=[], image=image)
        push_registry, push_repo = resolve_repository_name(self.name)
        try:
            results['actions'].append('Pushed image %s:%s' % (self.name, self.tag))
            headers = {}
            header = get_config_header(self.client, push_registry)
            if header:
                headers['X-Registry-Auth'] = header
            response = self.client._post_json(self.client._url('/images/{0}/push', self.name), data=None, headers=headers, stream=True, params={'tag': self.tag})
            self.client._raise_for_status(response)
            for line in self.client._stream_helper(response, decode=True):
                self.log(line, pretty_print=True)
                if line.get('errorDetail'):
                    raise Exception(line['errorDetail']['message'])
                status = line.get('status')
                if status == 'Pushing':
                    results['changed'] = True
        except Exception as exc:
            if 'unauthorized' in str(exc):
                if 'authentication required' in str(exc):
                    self.client.fail('Error pushing image %s/%s:%s - %s. Try logging into %s first.' % (push_registry, push_repo, self.tag, to_native(exc), push_registry))
                else:
                    self.client.fail('Error pushing image %s/%s:%s - %s. Does the repository exist?' % (push_registry, push_repo, self.tag, str(exc)))
            self.client.fail('Error pushing image %s:%s: %s' % (self.name, self.tag, to_native(exc)))
        return results