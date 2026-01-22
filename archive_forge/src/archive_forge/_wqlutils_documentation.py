from os_win._i18n import _
from os_win import exceptions
Returns the objects associated to an element as a list.

    :param conn: connection to be used to execute the query
    :param class_name: object's class type name to be retrieved
    :param element_instance_id: element class InstanceID
    :param element_uuid: UUID of the element
    :param fields: specific class attributes to be retrieved
    