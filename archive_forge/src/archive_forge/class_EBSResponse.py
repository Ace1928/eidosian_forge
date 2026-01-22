from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.backup.base import (
from libcloud.backup.types import BackupTargetType, BackupTargetJobStatusType
from libcloud.utils.iso8601 import parse_date
class EBSResponse(AWSGenericResponse):
    """
    Amazon EBS response class.
    """
    namespace = NS
    exceptions = {}
    xpath = 'Error'