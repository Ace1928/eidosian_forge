import abc
class DeviceNotFoundByNumberError(DeviceNotFoundError):
    """
    A :exc:`DeviceNotFoundError` indicating, that no :class:`Device` was found
    for a given device number.
    """

    def __init__(self, typ, number):
        DeviceNotFoundError.__init__(self, typ, number)

    @property
    def device_type(self):
        """
        The device type causing this error as string.  Either ``'char'`` or
        ``'block'``.
        """
        return self.args[0]

    @property
    def device_number(self):
        """
        The device number causing this error as integer.
        """
        return self.args[1]

    def __str__(self):
        return 'No {0.device_type} device with number {0.device_number}'.format(self)