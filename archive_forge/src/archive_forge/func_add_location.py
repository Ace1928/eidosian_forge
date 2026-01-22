import testtools
def add_location(self, *args, **kwargs):
    resp = self.controller.add_location(*args, **kwargs)
    self._assertRequestId(resp)