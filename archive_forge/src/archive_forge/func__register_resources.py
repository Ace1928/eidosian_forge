from stevedore import extension
from heat.common import pluginutils
from heat.engine import clients
from heat.engine import environment
from heat.engine import plugin_manager
def _register_resources(env, type_pairs):
    for res_name, res_class in type_pairs:
        env.register_class(res_name, res_class)