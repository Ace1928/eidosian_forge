from libcloud.common.types import LibcloudError
class ContainerIsNotEmptyError(ContainerError):
    error_type = 'ContainerIsNotEmptyError'