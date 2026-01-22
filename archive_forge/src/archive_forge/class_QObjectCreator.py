import os.path
from .exceptions import NoSuchWidgetError, WidgetPluginError
class QObjectCreator(object):

    def __init__(self, creatorPolicy):
        self._cpolicy = creatorPolicy
        self._cwFilters = []
        self._modules = self._cpolicy.createQtGuiWidgetsWrappers()
        for plugindir in widgetPluginPath:
            try:
                plugins = os.listdir(plugindir)
            except:
                plugins = []
            for filename in plugins:
                if not filename.endswith('.py'):
                    continue
                filename = os.path.join(plugindir, filename)
                plugin_globals = {'MODULE': MODULE, 'CW_FILTER': CW_FILTER, 'MATCH': MATCH, 'NO_MATCH': NO_MATCH}
                plugin_locals = {}
                if self.load_plugin(filename, plugin_globals, plugin_locals):
                    pluginType = plugin_locals['pluginType']
                    if pluginType == MODULE:
                        modinfo = plugin_locals['moduleInformation']()
                        self._modules.append(self._cpolicy.createModuleWrapper(*modinfo))
                    elif pluginType == CW_FILTER:
                        self._cwFilters.append(plugin_locals['getFilter']())
                    else:
                        raise WidgetPluginError('Unknown plugin type of %s' % filename)
        self._customWidgets = self._cpolicy.createCustomWidgetLoader()
        self._modules.append(self._customWidgets)

    def createQObject(self, classname, *args, **kwargs):
        factory = self.findQObjectType(classname)
        if factory is None:
            parts = classname.split('.')
            if len(parts) > 1:
                factory = self.findQObjectType(parts[0])
                if factory is not None:
                    for part in parts[1:]:
                        factory = getattr(factory, part, None)
                        if factory is None:
                            break
            if factory is None:
                raise NoSuchWidgetError(classname)
        return self._cpolicy.instantiate(factory, *args, **kwargs)

    def invoke(self, rname, method, args=()):
        return self._cpolicy.invoke(rname, method, args)

    def findQObjectType(self, classname):
        for module in self._modules:
            w = module.search(classname)
            if w is not None:
                return w
        return None

    def getSlot(self, obj, slotname):
        return self._cpolicy.getSlot(obj, slotname)

    def asString(self, s):
        return self._cpolicy.asString(s)

    def addCustomWidget(self, widgetClass, baseClass, module):
        for cwFilter in self._cwFilters:
            match, result = cwFilter(widgetClass, baseClass, module)
            if match:
                widgetClass, baseClass, module = result
                break
        self._customWidgets.addCustomWidget(widgetClass, baseClass, module)

    @staticmethod
    def load_plugin(filename, plugin_globals, plugin_locals):
        """ Load the plugin from the given file.  Return True if the plugin was
        loaded, or False if it wanted to be ignored.  Raise an exception if
        there was an error.
        """
        plugin = open(filename)
        try:
            exec(plugin.read(), plugin_globals, plugin_locals)
        except ImportError:
            return False
        except Exception as e:
            raise WidgetPluginError('%s: %s' % (e.__class__, str(e)))
        finally:
            plugin.close()
        return True