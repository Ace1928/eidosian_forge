from os_ken.controller import event as os_ken_event
from os_ken.controller import handler
class EventRowUpdatedBase(EventRowUpdate):

    def __init__(self, ev):
        super(EventRowUpdatedBase, self).__init__(ev.system_id, ev.table, ev.old, ev.new)