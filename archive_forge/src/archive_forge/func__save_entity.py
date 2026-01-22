from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def _save_entity(self, entity):
    """
        Updates an existing entity
        :param entity: The entity to save
        :return: The updated entity
        """
    try:
        entity.save()
    except BambouHTTPError as error:
        self.module.fail_json(msg='Unable to update entity: {0}'.format(error))
    self.result['changed'] = True
    return entity