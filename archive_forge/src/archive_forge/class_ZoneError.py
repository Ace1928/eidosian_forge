from libcloud.common.types import LibcloudError
class ZoneError(LibcloudError):
    error_type = 'ZoneError'
    kwargs = ('zone_id',)

    def __init__(self, value, driver, zone_id):
        self.zone_id = zone_id
        super().__init__(value=value, driver=driver)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<{} in {}, zone_id={}, value={}>'.format(self.error_type, repr(self.driver), self.zone_id, self.value)