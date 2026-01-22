from libcloud.common.types import LibcloudError
class RecordAlreadyExistsError(RecordError):
    error_type = 'RecordAlreadyExistsError'