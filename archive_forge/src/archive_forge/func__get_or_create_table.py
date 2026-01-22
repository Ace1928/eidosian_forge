from collections import namedtuple
from time import sleep, time
from typing import Any, Dict
from kombu.utils.url import _parse_url as parse_url
from celery.exceptions import ImproperlyConfigured
from celery.utils.log import get_logger
from .base import KeyValueStoreBackend
def _get_or_create_table(self):
    """Create table if not exists, otherwise return the description."""
    table_schema = self._get_table_schema()
    try:
        return self._client.describe_table(TableName=self.table_name)
    except ClientError as e:
        error_code = e.response['Error'].get('Code', 'Unknown')
        if error_code == 'ResourceNotFoundException':
            table_description = self._client.create_table(**table_schema)
            logger.info('DynamoDB Table {} did not exist, creating.'.format(self.table_name))
            self._wait_for_table_status('ACTIVE')
            logger.info('DynamoDB Table {} is now available.'.format(self.table_name))
            return table_description
        else:
            raise e