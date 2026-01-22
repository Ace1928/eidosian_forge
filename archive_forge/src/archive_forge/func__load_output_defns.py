import itertools
from heat.common import exception
from heat.engine import attributes
from heat.engine import status
def _load_output_defns(self):
    self._output_defns = self._template.outputs(self)