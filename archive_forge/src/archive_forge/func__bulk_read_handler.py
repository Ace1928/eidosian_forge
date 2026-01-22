import ssl
import socket
from os_ken import cfg
from os_ken.base import app_manager
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.services.protocols.ovsdb import client
from os_ken.services.protocols.ovsdb import event
from os_ken.controller import handler
def _bulk_read_handler(self, ev):
    results = []

    def done(gt, *args, **kwargs):
        if gt in self.threads:
            self.threads.remove(gt)
        results.append(gt.wait())
    threads = []
    for c in self._clients.values():
        gt = hub.spawn(c.read_request_handler, ev, bulk=True)
        threads.append(gt)
        self.threads.append(gt)
        gt.link(done)
    hub.joinall(threads)
    rep = event.EventReadReply(None, results)
    self.reply_to_request(ev, rep)