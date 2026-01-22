from libcloud.common.types import LibcloudError
class ZoneDoesNotExistError(ZoneError):
    error_type = 'ZoneDoesNotExistError'