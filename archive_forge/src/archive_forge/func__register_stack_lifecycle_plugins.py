from stevedore import extension
from heat.common import pluginutils
from heat.engine import clients
from heat.engine import environment
from heat.engine import plugin_manager
def _register_stack_lifecycle_plugins(env, type_pairs):
    for stack_lifecycle_name, stack_lifecycle_class in type_pairs:
        env.register_stack_lifecycle_plugin(stack_lifecycle_name, stack_lifecycle_class)