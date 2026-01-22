import concurrent.futures
import urllib.parse
import keystoneauth1.exceptions
from openstack.cloud import _utils
from openstack import exceptions
from openstack.object_store.v1._proxy import Proxy
def _wait_for_futures(self, futures, raise_on_error=True):
    """Collect results or failures from a list of running future tasks."""
    results = []
    retries = []
    for completed in concurrent.futures.as_completed(futures):
        try:
            result = completed.result()
            exceptions.raise_from_response(result)
            results.append(result)
        except (keystoneauth1.exceptions.RetriableConnectionFailure, exceptions.HttpException) as e:
            error_text = 'Exception processing async task: {}'.format(str(e))
            if raise_on_error:
                self.log.exception(error_text)
                raise
            else:
                self.log.debug(error_text)
            retries.append(completed.result())
    return (results, retries)