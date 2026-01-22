import eventlet
from oslo_config import cfg
import oslo_messaging
from oslo_messaging.rpc import dispatcher
from osprofiler import profiler
from heat.common import context
class RequestContextSerializer(oslo_messaging.Serializer):

    def __init__(self, base):
        self._base = base

    def serialize_entity(self, ctxt, entity):
        if not self._base:
            return entity
        return self._base.serialize_entity(ctxt, entity)

    def deserialize_entity(self, ctxt, entity):
        if not self._base:
            return entity
        return self._base.deserialize_entity(ctxt, entity)

    @staticmethod
    def serialize_context(ctxt):
        _context = ctxt.to_dict()
        prof = profiler.get()
        if prof:
            trace_info = {'hmac_key': prof.hmac_key, 'base_id': prof.get_base_id(), 'parent_id': prof.get_id()}
            _context.update({'trace_info': trace_info})
        return _context

    @staticmethod
    def deserialize_context(ctxt):
        trace_info = ctxt.pop('trace_info', None)
        if trace_info:
            profiler.init(**trace_info)
        return context.RequestContext.from_dict(ctxt)