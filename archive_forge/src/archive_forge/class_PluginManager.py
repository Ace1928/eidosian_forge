import types
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_trace_api
class PluginManager(object):

    def __init__(self, main_debugger):
        self.plugins = load_plugins()
        self.active_plugins = []
        self.main_debugger = main_debugger
        self.rebind_methods()

    def add_breakpoint(self, func_name, *args, **kwargs):
        for plugin in self.plugins:
            if hasattr(plugin, func_name):
                func = getattr(plugin, func_name)
                result = func(self, *args, **kwargs)
                if result:
                    self.activate(plugin)
                    return result
        return None

    def activate(self, plugin):
        if plugin not in self.active_plugins:
            self.active_plugins.append(plugin)
            self.rebind_methods()

    def rebind_methods(self):
        if len(self.active_plugins) == 0:
            self.bind_functions(pydevd_trace_api, getattr, pydevd_trace_api)
        elif len(self.active_plugins) == 1:
            self.bind_functions(pydevd_trace_api, getattr, self.active_plugins[0])
        else:
            self.bind_functions(pydevd_trace_api, create_dispatch, self.active_plugins)

    def bind_functions(self, interface, function_factory, arg):
        for name in dir(interface):
            func = function_factory(arg, name)
            if type(func) == types.FunctionType:
                bind_func_to_method(func, self, name)