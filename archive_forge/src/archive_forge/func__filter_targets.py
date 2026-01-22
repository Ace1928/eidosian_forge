from functools import wraps
from pyomo.common.collections import ComponentMap
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import unique_component_name
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core.base.external import ExternalFunction
from pyomo.core import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import (
from pyomo.network import Port
from weakref import ref as weakref_ref
def _filter_targets(self, instance):
    targets = self._config.targets
    if targets is None:
        targets = (instance,)

    def _filter_inactive(targets):
        for t in targets:
            if not t.active:
                self.logger.warning(f'GDP.{self.transformation_name} transformation passed a deactivated target ({t.name}). Skipping.')
            else:
                yield t
    return list(_filter_inactive(targets))