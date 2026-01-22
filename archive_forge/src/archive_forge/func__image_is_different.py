from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def _image_is_different(self, image, container):
    if image and image.get('Id'):
        if container and container.image:
            if image.get('Id') != container.image:
                self.diff_tracker.add('image', parameter=image.get('Id'), active=container.image)
                return True
    return False