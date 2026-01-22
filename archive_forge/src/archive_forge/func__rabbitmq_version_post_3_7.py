from __future__ import absolute_import, division, print_function
import ansible_collections.community.rabbitmq.plugins.module_utils.version as Version
import json
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.collections import count
def _rabbitmq_version_post_3_7(self, fail_on_error=False):
    """Use the JSON formatter to get a machine readable output of the version.

        At this point we do not know which RabbitMQ server version we are dealing with and which
        version of `rabbitmqctl` we are using, so we will try to use the JSON formatter and see
        what happens. In some versions of
        """

    def int_list_to_str(ints):
        return ''.join([chr(i) for i in ints])
    rc, output, err = self._exec(['status', '--formatter', 'json'], check_rc=False)
    if rc != 0:
        return self._fail(msg='Could not parse the version of the RabbitMQ server, because `rabbitmqctl status` returned no output.', stop_execution=fail_on_error)
    try:
        status_json = json.loads(output)
        if 'rabbitmq_version' in status_json:
            return Version.StrictVersion(status_json['rabbitmq_version'])
        for application in status_json.get('running_applications', list()):
            if application[0] == 'rabbit':
                if isinstance(application[1][0], int):
                    return Version.StrictVersion(int_list_to_str(application[2]))
                else:
                    return Version.StrictVersion(application[1])
        return self._fail(msg='Could not find RabbitMQ version of `rabbitmqctl status` command.', stop_execution=fail_on_error)
    except ValueError as e:
        return self._fail(msg='Could not parse output of `rabbitmqctl status` as JSON: {exc}.'.format(exc=repr(e)), stop_execution=fail_on_error)