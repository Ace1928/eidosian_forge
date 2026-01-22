importing any other Kivy modules. Ideally, this means setting them right at
from collections import OrderedDict
from os import environ
from os.path import exists
from weakref import ref
from kivy import kivy_config_fn
from kivy.compat import PY2, string_types
from kivy.logger import Logger, logger_config_update
from kivy.utils import platform
@staticmethod
def _register_named_property(name, widget_ref, *largs):
    """ Called by the ConfigParserProperty to register a property which
        was created with a config name instead of a config object.

        When a ConfigParser with this name is later created, the properties
        are then notified that this parser now exists so they can use it.
        If the parser already exists, the property is notified here. See
        :meth:`~kivy.properties.ConfigParserProperty.set_config`.

        :Parameters:
            `name`: a non-empty string
                The name of the ConfigParser that is associated with the
                property. See :attr:`name`.
            `widget_ref`: 2-tuple.
                The first element is a reference to the widget containing the
                property, the second element is the name of the property. E.g.:

                    class House(Widget):
                        address = ConfigParserProperty('', 'info', 'street',
                            'directory')

                Then, the first element is a ref to a House instance, and the
                second is `'address'`.
        """
    configs = ConfigParser._named_configs
    try:
        config, props = configs[name]
    except KeyError:
        configs[name] = (None, [widget_ref])
        return
    props.append(widget_ref)
    if config:
        config = config()
    widget = widget_ref[0]()
    if config and widget:
        widget.property(widget_ref[1]).set_config(config)