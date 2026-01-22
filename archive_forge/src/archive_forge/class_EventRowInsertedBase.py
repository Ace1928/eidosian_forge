from os_ken.controller import event as os_ken_event
from os_ken.controller import handler
class EventRowInsertedBase(EventRowInsert):

    def __init__(self, ev):
        super(EventRowInsertedBase, self).__init__(ev.system_id, ev.table, ev.row)