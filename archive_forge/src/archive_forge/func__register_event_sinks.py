from stevedore import extension
from heat.common import pluginutils
from heat.engine import clients
from heat.engine import environment
from heat.engine import plugin_manager
def _register_event_sinks(env, type_pairs):
    for sink_name, sink_class in type_pairs:
        env.register_event_sink(sink_name, sink_class)