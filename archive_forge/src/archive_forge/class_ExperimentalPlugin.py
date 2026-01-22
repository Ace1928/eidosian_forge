class ExperimentalPlugin:
    """A marker class used to annotate a plugin as experimental.

    Experimental plugins are hidden from users by default. The plugin will only
    be enabled for a user if the user has specified the plugin with the
    experimentalPlugin query parameter in the URL.

    The marker class can annotate either TBPlugin or TBLoader instances, whichever
    is most convenient.

    Typical usage is to create a new class that inherits from both an existing
    TBPlugin/TBLoader class and this marker class. For example:

    class ExperimentalGraphsPlugin(
        graphs_plugin.GraphsPlugin,
        experimental_plugin.ExperimentalPlugin,
    ):
        pass


    class ExperimentalDebuggerPluginLoader(
        debugger_plugin_loader.DebuggerPluginLoader,
        experimental_plugin.ExperimentalPlugin
    ):
        pass

    Note: This class is itself an experimental mechanism and is subject to
    modification or removal without warning.
    """
    pass