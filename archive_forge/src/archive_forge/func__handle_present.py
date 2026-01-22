from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def _handle_present(self):
    """
        Handles the Ansible task when the state is set to present
        """
    self.entity = self._find_entity(entity_id=self.entity_id, entity_class=self.entity_class, match_filter=self.match_filter, properties=self.properties, entity_fetcher=self.entity_fetcher)
    if self.entity_fetcher is not None and self.entity_fetcher.relationship == 'member' and (not self.entity):
        self.module.fail_json(msg='Trying to assign an entity that does not exist')
    elif self.entity_fetcher is not None and self.entity_fetcher.relationship == 'member' and self.entity:
        if not self._is_member(entity_fetcher=self.entity_fetcher, entity=self.entity):
            if self.module.check_mode:
                self.result['changed'] = True
            else:
                self._assign_member(entity_fetcher=self.entity_fetcher, entity=self.entity, entity_class=self.entity_class, parent=self.parent, set_output=True)
    elif self.entity_fetcher is not None and self.entity_fetcher.relationship in ['child', 'root'] and (not self.entity):
        if self.module.check_mode:
            self.result['changed'] = True
        else:
            self.entity = self._create_entity(entity_class=self.entity_class, parent=self.parent, properties=self.properties)
            self.result['id'] = self.entity.id
            self.result['entities'].append(self.entity.to_dict())
        if self.children:
            for child in self.children:
                self._handle_child(child=child, parent=self.entity)
    elif self.entity:
        changed = self._has_changed(entity=self.entity, properties=self.properties)
        if self.module.check_mode:
            self.result['changed'] = changed
        elif changed:
            self.entity = self._save_entity(entity=self.entity)
            self.result['id'] = self.entity.id
            self.result['entities'].append(self.entity.to_dict())
        else:
            self.result['id'] = self.entity.id
            self.result['entities'].append(self.entity.to_dict())
        if self.children:
            for child in self.children:
                self._handle_child(child=child, parent=self.entity)
    elif not self.module.check_mode:
        self.module.fail_json(msg='Invalid situation, verify parameters')