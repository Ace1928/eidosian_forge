import os
from twisted.spread import pb
class MaildirBroker(pb.Broker):

    def proto_getCollection(self, requestID, name, domain, password):
        collection = self._getCollection()
        if collection is None:
            self.sendError(requestID, 'permission denied')
        else:
            self.sendAnswer(requestID, collection)

    def getCollection(self, name, domain, password):
        if domain not in self.domains:
            return
        domain = self.domains[domain]
        if name in domain.dbm and domain.dbm[name] == password:
            return MaildirCollection(domain.userDirectory(name))