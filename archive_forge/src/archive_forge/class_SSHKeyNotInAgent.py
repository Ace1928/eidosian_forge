from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class SSHKeyNotInAgent(exceptions.Error):
    """Error when the SSH key is not in the SSH agent."""

    def __init__(self, identity_file):
        super(SSHKeyNotInAgent, self).__init__('SSH Key is not present in the SSH agent. Please run "ssh-add {}" to add it, and try again.'.format(identity_file))