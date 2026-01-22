from os_ken.controller import event as os_ken_event
from os_ken.controller import handler
class EventRowUpdate(os_ken_event.EventBase):

    def __init__(self, system_id, table, old, new):
        super(EventRowUpdate, self).__init__()
        self.system_id = system_id
        self.table = table
        self.old = old
        self.new = new
        self.event_type = 'Updated'

    def __str__(self):
        return '%s<system_id=%s table=%s, uuid=%s>' % (self.__class__.__name__, self.system_id, self.table, self.old['_uuid'])