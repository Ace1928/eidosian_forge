from suds.cache import Cache, NoCache
from suds.properties import *
from suds.store import DocumentStore, defaultDocumentStore
from suds.transport import Transport
from suds.wsse import Security
from suds.xsd.doctor import Doctor
class TpLinker(AutoLinker):
    """
    Transport (auto) linker used to manage linkage between
    transport objects Properties and those Properties that contain them.
    """

    def updated(self, properties, prev, next):
        if isinstance(prev, Transport):
            tp = Unskin(prev.options)
            properties.unlink(tp)
        if isinstance(next, Transport):
            tp = Unskin(next.options)
            properties.link(tp)