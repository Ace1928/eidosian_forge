from __future__ import (absolute_import, division, print_function)
import traceback
from collections import defaultdict
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _update_label_assignments(self, entity, name, label_obj_type):
    objs_service = getattr(self._connection.system_service(), '%s_service' % name)()
    if self._module.params[name] is not None:
        objs = self._connection.follow_link(getattr(entity, name))
        objs_names = defaultdict(list)
        for obj in objs:
            labeled_entity = objs_service.service(obj.id).get()
            if self._module.params['cluster'] is None:
                objs_names[labeled_entity.name].append(obj.id)
            elif self._connection.follow_link(labeled_entity.cluster).name == self._module.params['cluster']:
                objs_names[labeled_entity.name].append(obj.id)
        for obj in self._module.params[name]:
            if obj not in objs_names:
                for obj_id in objs_service.list(search='name=%s and cluster=%s' % (obj, self._module.params['cluster'])):
                    label_service = getattr(self._service.service(entity.id), '%s_service' % name)()
                    if not self._module.check_mode:
                        label_service.add(**{name[:-1]: label_obj_type(id=obj_id.id)})
                    self.changed = True
        for obj in objs_names:
            if obj not in self._module.params[name]:
                label_service = getattr(self._service.service(entity.id), '%s_service' % name)()
                if not self._module.check_mode:
                    for obj_id in objs_names[obj]:
                        label_service.service(obj_id).remove()
                self.changed = True