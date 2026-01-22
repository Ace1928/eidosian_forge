from gi.repository import GObject
import dbus.service
def ExportedGObject__init__(self, conn=None, object_path=None, **kwargs):
    """Initialize an exported GObject.

    :Parameters:
        `conn` : dbus.connection.Connection
            The D-Bus connection or bus
        `object_path` : str
            The object path at which to register this object.
    :Keywords:
        `bus_name` : dbus.service.BusName
            A bus name to be held on behalf of this object, or None.
        `gobject_properties` : dict
            GObject properties to be set on the constructed object.

            Any unrecognised keyword arguments will also be interpreted
            as GObject properties.
        """
    bus_name = kwargs.pop('bus_name', None)
    gobject_properties = kwargs.pop('gobject_properties', None)
    if gobject_properties is not None:
        kwargs.update(gobject_properties)
    GObject.GObject.__init__(self, **kwargs)
    dbus.service.Object.__init__(self, conn=conn, object_path=object_path, bus_name=bus_name)