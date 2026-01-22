from Xlib import X
from Xlib.protocol import rq, structs
class SetClipRectangles(rq.Request):
    _request = rq.Struct(rq.Opcode(59), rq.Set('ordering', 1, (X.Unsorted, X.YSorted, X.YXSorted, X.YXBanded)), rq.RequestLength(), rq.GC('gc'), rq.Int16('x_origin'), rq.Int16('y_origin'), rq.List('rectangles', structs.Rectangle))