import itertools
from heat.common import exception
from heat.engine import attributes
from heat.engine import status
def _res_data(self):
    assert self._resource_data is not None, 'Resource data not available'
    return self._resource_data