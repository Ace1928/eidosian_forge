from os_ken.controller import event as os_ken_event
from os_ken.controller import handler
class EventModifyRequest(os_ken_event.EventRequestBase):
    """ Dispatch a modify function to OVSDB

    `func` must be a callable that accepts an insert fucntion and the
    IDL.tables object. It can then modify the tables as needed. For inserts,
    specify a UUID for each insert, and return a tuple of the temporary
    UUID's. The execution of `func` will be wrapped in a single transaction
    and the reply will include a dict of temporary UUID to real UUID mappings.

    e.g.

        new_port_uuid = uuid.uuid4()

        def modify(tables, insert):
            bridges = tables['Bridge'].rows
            bridge = None
            for b in bridges:
                if b.name == 'my-bridge':
                    bridge = b

            if not bridge:
                return

            port = insert('Port', new_port_uuid)

            bridge.ports = bridge.ports + [port]

            return (new_port_uuid, )

        request = EventModifyRequest(system_id, modify)
        reply = send_request(request)

        port_uuid = reply.insert_uuids[new_port_uuid]
    """

    def __init__(self, system_id, func):
        super(EventModifyRequest, self).__init__()
        self.dst = 'OVSDB'
        self.system_id = system_id
        self.func = func

    def __str__(self):
        return '%s<system_id=%s>' % (self.__class__.__name__, self.system_id)