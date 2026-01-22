import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
class ACLFormatter(formatter.EntityFormatter):
    columns = ('Operation Type', 'Project Access', 'Users', 'Created', 'Updated')

    def _get_formatted_data(self):
        created = self.created.isoformat() if self.created else None
        updated = self.updated.isoformat() if self.updated else None
        data = (self.operation_type, self.project_access, self.users, created, updated, self.acl_ref)
        return data