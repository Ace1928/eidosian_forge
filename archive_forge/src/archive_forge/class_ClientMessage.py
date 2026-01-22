from Xlib import X
from Xlib.protocol import rq
class ClientMessage(rq.Event):
    _code = X.ClientMessage
    _fields = rq.Struct(rq.Card8('type'), rq.Format('data', 1), rq.Card16('sequence_number'), rq.Window('window'), rq.Card32('client_type'), rq.FixedPropertyData('data', 20))