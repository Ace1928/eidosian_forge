from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
from ansible_collections.community.docker.plugins.module_utils._platform import (
class ImagePuller(DockerBaseClass):

    def __init__(self, client):
        super(ImagePuller, self).__init__()
        self.client = client
        self.check_mode = self.client.check_mode
        parameters = self.client.module.params
        self.name = parameters['name']
        self.tag = parameters['tag']
        self.platform = parameters['platform']
        self.pull_mode = parameters['pull']
        if is_image_name_id(self.name):
            self.client.fail('Cannot pull an image by ID')
        if not is_valid_tag(self.tag, allow_empty=True):
            self.client.fail('"{0}" is not a valid docker tag!'.format(self.tag))
        repo, repo_tag = parse_repository_tag(self.name)
        if repo_tag:
            self.name = repo
            self.tag = repo_tag

    def pull(self):
        image = self.client.find_image(name=self.name, tag=self.tag)
        results = dict(changed=False, actions=[], image=image or {}, diff=dict(before=image_info(image), after=image_info(image)))
        if image and self.pull_mode == 'not_present':
            if self.platform is None:
                return results
            host_info = self.client.info()
            wanted_platform = normalize_platform_string(self.platform, daemon_os=host_info.get('OSType'), daemon_arch=host_info.get('Architecture'))
            image_platform = compose_platform_string(os=image.get('Os'), arch=image.get('Architecture'), variant=image.get('Variant'), daemon_os=host_info.get('OSType'), daemon_arch=host_info.get('Architecture'))
            if compare_platform_strings(wanted_platform, image_platform):
                return results
        results['actions'].append('Pulled image %s:%s' % (self.name, self.tag))
        if self.check_mode:
            results['changed'] = True
            results['diff']['after'] = image_info(dict(Id='unknown'))
        else:
            results['image'], not_changed = self.client.pull_image(self.name, tag=self.tag, platform=self.platform)
            results['changed'] = not not_changed
            results['diff']['after'] = image_info(results['image'])
        return results