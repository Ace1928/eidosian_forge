from urllib import parse
from cinderclient import api_versions
from cinderclient import base
@api_versions.wraps('3.11')
def unset_keys(self, keys):
    """Unset specs on a group type.

        :param type_id: The :class:`GroupType` to unset spec on
        :param keys: A list of keys to be unset
        """
    for k in keys:
        resp = self.manager._delete('/group_types/%s/group_specs/%s' % (base.getid(self), k))
        if resp:
            return resp