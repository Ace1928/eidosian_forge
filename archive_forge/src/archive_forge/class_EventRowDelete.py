from os_ken.controller import event as os_ken_event
from os_ken.controller import handler
class EventRowDelete(EventRowBase):

    def __init__(self, system_id, table, row):
        super(EventRowDelete, self).__init__(system_id, table, row, 'Deleted')