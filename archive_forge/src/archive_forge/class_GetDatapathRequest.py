import numbers
from os_ken.controller import event
class GetDatapathRequest(_RequestBase):

    def __init__(self, dpid=None):
        assert dpid is None or isinstance(dpid, numbers.Integral)
        super(GetDatapathRequest, self).__init__()
        self.dpid = dpid