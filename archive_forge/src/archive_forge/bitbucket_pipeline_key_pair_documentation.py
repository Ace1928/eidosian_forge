from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.source_control.bitbucket import BitbucketHelper

    Retrieves an existing ssh key pair from repository
    specified in module param `repository`

    :param module: instance of the :class:`AnsibleModule`
    :param bitbucket: instance of the :class:`BitbucketHelper`
    :return: existing key pair or None if not found
    :rtype: dict or None

    Return example::

        {
            "public_key": "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQ...2E8HAeT",
            "type": "pipeline_ssh_key_pair"
        }
    