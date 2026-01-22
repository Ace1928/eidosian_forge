from libcloud.common.types import LibcloudError
class RecordDoesNotExistError(RecordError):
    error_type = 'RecordDoesNotExistError'