import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
class CAFormatter(formatter.EntityFormatter):
    columns = ('CA href', 'Name', 'Description', 'Created', 'Updated', 'Status', 'Plugin Name', 'Plugin CA ID', 'Expiration')

    def _get_formatted_data(self):
        created = self.created.isoformat() if self.created else None
        updated = self.updated.isoformat() if self.updated else None
        expiration = self.expiration.isoformat() if self.expiration else None
        data = (self.ca_ref, self.name, self.description, created, updated, self.status, self.plugin_name, self.plugin_ca_id, expiration)
        return data