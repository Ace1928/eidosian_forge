from Xlib.protocol import request
from Xlib.xobject import resource, cursor
class Fontable(resource.Resource):
    __fontable__ = resource.Resource.__resource__

    def query(self):
        return request.QueryFont(display=self.display, font=self.id)

    def query_text_extents(self, string):
        return request.QueryTextExtents(display=self.display, font=self.id, string=string)