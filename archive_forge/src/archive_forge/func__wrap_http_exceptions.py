import contextlib
import logging
import keystoneauth1.exceptions.http as ks_exceptions
import osc_lib.exceptions as exceptions
import simplejson as json
from osc_placement import version
@contextlib.contextmanager
def _wrap_http_exceptions():
    """Reraise osc-lib exceptions with detailed messages."""
    try:
        yield
    except ks_exceptions.HttpError as exc:
        if 400 <= exc.http_status < 500:
            detail = json.loads(exc.response.content)['errors'][0]['detail']
            msg = detail.split('\n')[-1].strip()
            exc_class = _http_error_to_exc.get(exc.http_status, exceptions.CommandError)
            raise exc_class(exc.http_status, msg) from exc
        else:
            raise