import logging, sys
from weakref import ref as weakref_ref
from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import ComponentMap
from pyomo.common.deprecation import RenamedClass
from pyomo.common.formatting import tabular_writer
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import unique_component_name, NOTSET
from pyomo.common.numeric_types import value
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import IndexedComponent, UnindexedComponent_set
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core.expr import identify_variables
from pyomo.core.base.label import alphanum_label_from_name
from pyomo.network.util import create_var, tighten_var_domain
@staticmethod
def Extensive(port, name, index_set, include_splitfrac=None, write_var_sum=True):
    """Arc Expansion procedure for extensive variable properties

        This procedure is the rule to use when variable quantities should
        be conserved; that is, split for outlets and combined for inlets.

        This will first go through every destination of the port (i.e.,
        arcs whose source is this Port) and create a new variable on the
        arc's expanded block of the same index as the current variable
        being processed to store the amount of the variable that flows
        over the arc.  For ports that have multiple outgoing arcs, this
        procedure will create a single splitfrac variable on the arc's
        expanded block as well. Then it will generate constraints for
        the new variable that relate it to the port member variable
        using the split fraction, ensuring that all extensive variables
        in the Port are split using the same ratio.  The generation of
        the split fraction variable and constraint can be suppressed by
        setting the `include_splitfrac` argument to `False`.

        Once all arc-specific variables are created, this
        procedure will create the "balancing constraint" that ensures
        that the sum of all the new variables equals the original port
        member variable. This constraint can be suppressed by setting
        the `write_var_sum` argument to `False`; in which case, a single
        constraint will be written that states the sum of the split
        fractions equals 1.

        Finally, this procedure will go through every source for this
        port and create a new arc variable (unless it already exists),
        before generating the balancing constraint that ensures the sum
        of all the incoming new arc variables equals the original port
        variable.

        Model simplifications:

            If the port has a 1-to-1 connection on either side, it will not
            create the new variables and instead write a simple equality
            constraint for that side.

            If the outlet side is not 1-to-1 but there is only one outlet,
            it will not create a splitfrac variable or write the split
            constraint, but it will still write the outsum constraint
            which will be a simple equality.

            If the port only contains a single Extensive variable, the
            splitfrac variables and the splitting constraints will
            be skipped since they will be unnecessary. However, they
            can be still be included by passing `include_splitfrac=True`.

        .. note::
            If split fractions are skipped, the `write_var_sum=False`
            option is not allowed.

        """
    port_parent = port.parent_block()
    out_vars = Port._Split(port, name, index_set, include_splitfrac=include_splitfrac, write_var_sum=write_var_sum)
    in_vars = Port._Combine(port, name, index_set)