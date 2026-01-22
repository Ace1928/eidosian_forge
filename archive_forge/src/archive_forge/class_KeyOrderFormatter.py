import abc
import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
class KeyOrderFormatter(formatter.EntityFormatter):
    columns = ('Order href', 'Type', 'Container href', 'Secret href', 'Created', 'Status', 'Error code', 'Error message')

    def _get_formatted_data(self):
        created = self.created.isoformat() if self.created else None
        data = (self.order_ref, 'Key', 'N/A', self.secret_ref, created, self.status, self.error_status_code, self.error_reason)
        return data