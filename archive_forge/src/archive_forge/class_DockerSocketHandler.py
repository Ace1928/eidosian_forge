from __future__ import (absolute_import, division, print_function)
from ansible.compat import selectors
from ansible_collections.community.docker.plugins.module_utils.socket_handler import (
class DockerSocketHandler(DockerSocketHandlerBase):

    def __init__(self, display, sock, log=None, container=None):
        super(DockerSocketHandler, self).__init__(sock, selectors, log=lambda msg: display.vvvv(msg, host=container))