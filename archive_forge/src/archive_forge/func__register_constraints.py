from stevedore import extension
from heat.common import pluginutils
from heat.engine import clients
from heat.engine import environment
from heat.engine import plugin_manager
def _register_constraints(env, type_pairs):
    for constraint_name, constraint in type_pairs:
        env.register_constraint(constraint_name, constraint)