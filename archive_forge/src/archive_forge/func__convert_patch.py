from openstack import exceptions
from openstack import resource
def _convert_patch(self, patch):
    converted = super(AcceleratorRequest, self)._convert_patch(patch)
    converted = {self.id: converted}
    return converted