from Xlib import X
from Xlib.protocol import rq
class EnableContext(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(5), rq.RequestLength(), rq.Card32('context'))
    _reply = rq.Struct(rq.Pad(1), rq.Card8('category'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card8('element_header'), rq.Bool('client_swapped'), rq.Pad(2), rq.Card32('id_base'), rq.Card32('server_time'), rq.Card32('recorded_sequence_number'), rq.Pad(8), RawField('data'))

    def __init__(self, callback, *args, **keys):
        self._callback = callback
        rq.ReplyRequest.__init__(self, *args, **keys)

    def _parse_response(self, data):
        r, d = self._reply.parse_binary(data, self._display)
        self._callback(r)
        if r.category == StartOfData:
            self.sequence_number = r.sequence_number
        if r.category == EndOfData:
            self._response_lock.acquire()
            self._data = r
            self._response_lock.release()
        else:
            self._display.sent_requests.insert(0, self)