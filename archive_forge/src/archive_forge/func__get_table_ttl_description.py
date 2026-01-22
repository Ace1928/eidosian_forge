from collections import namedtuple
from time import sleep, time
from typing import Any, Dict
from kombu.utils.url import _parse_url as parse_url
from celery.exceptions import ImproperlyConfigured
from celery.utils.log import get_logger
from .base import KeyValueStoreBackend
def _get_table_ttl_description(self):
    try:
        description = self._client.describe_time_to_live(TableName=self.table_name)
    except ClientError as e:
        error_code = e.response['Error'].get('Code', 'Unknown')
        error_message = e.response['Error'].get('Message', 'Unknown')
        logger.error('Error describing Time to Live on DynamoDB table {table}: {code}: {message}'.format(table=self.table_name, code=error_code, message=error_message))
        raise e
    return description