from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.backup.base import (
from libcloud.backup.types import BackupTargetType, BackupTargetJobStatusType
from libcloud.utils.iso8601 import parse_date
def _to_jobs(self, data):
    xpath = 'DescribeSnapshotsResponse/snapshotSet/item'
    return [self._to_job(el) for el in findall(element=data, xpath=xpath, namespace=NS)]