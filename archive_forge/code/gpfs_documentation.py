from os_brick.i18n import _
from os_brick.initiator.connectors import local
from os_brick import utils
Connect to a volume.

        :param connection_properties: The dictionary that describes all
                                      of the target volume attributes.
               connection_properties must include:
               device_path - path to the volume to be connected
        :type connection_properties: dict
        :returns: dict
        