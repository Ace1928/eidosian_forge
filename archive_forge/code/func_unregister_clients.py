from Xlib import X
from Xlib.protocol import rq
def unregister_clients(self, context, clients):
    UnregisterClients(display=self.display, opcode=self.display.get_extension_major(extname), context=context, clients=clients)