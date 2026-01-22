from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.source_control.bitbucket import BitbucketHelper

    Search for an existing deploy key on Bitbucket
    with the label specified in module param `label`

    :param module: instance of the :class:`AnsibleModule`
    :param bitbucket: instance of the :class:`BitbucketHelper`
    :return: existing deploy key or None if not found
    :rtype: dict or None

    Return example::

        {
            "id": 123,
            "label": "mykey",
            "created_on": "2019-03-23T10:15:21.517377+00:00",
            "key": "ssh-rsa AAAAB3NzaC1yc2EAAAADA...AdkTg7HGqL3rlaDrEcWfL7Lu6TnhBdq5",
            "type": "deploy_key",
            "comment": "",
            "last_used": None,
            "repository": {
                "links": {
                    "self": {
                        "href": "https://api.bitbucket.org/2.0/repositories/mleu/test"
                    },
                    "html": {
                        "href": "https://bitbucket.org/mleu/test"
                    },
                    "avatar": {
                        "href": "..."
                    }
                },
                "type": "repository",
                "name": "test",
                "full_name": "mleu/test",
                "uuid": "{85d08b4e-571d-44e9-a507-fa476535aa98}"
            },
            "links": {
                "self": {
                    "href": "https://api.bitbucket.org/2.0/repositories/mleu/test/deploy-keys/123"
                }
            },
        }
    