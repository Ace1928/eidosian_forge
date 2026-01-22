from decimal import Decimal
from boto.compat import filter, map
class ResponseResultList(Response):
    _ResultClass = ResponseElement

    def __init__(self, *args, **kw):
        setattr(self, self._action + 'Result', ElementList(self._ResultClass))
        super(ResponseResultList, self).__init__(*args, **kw)