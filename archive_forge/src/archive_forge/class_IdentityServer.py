import urllib.parse
from keystoneauth1 import discover
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import plugin
from keystoneclient.v3 import client as v3_client
from keystonemiddleware.auth_token import _auth
from keystonemiddleware.auth_token import _exceptions as ksm_exceptions
from keystonemiddleware.i18n import _
class IdentityServer(object):
    """Base class for operations on the Identity API server.

    The auth_token middleware needs to communicate with the Identity API server
    to validate tokens. This class encapsulates the data and methods to perform
    the operations.

    """

    def __init__(self, log, adap, include_service_catalog=None, requested_auth_version=None, requested_auth_interface=None):
        self._LOG = log
        self._adapter = adap
        self._include_service_catalog = include_service_catalog
        self._requested_auth_version = requested_auth_version
        self._requested_auth_interface = requested_auth_interface
        self._request_strategy_obj = None

    @property
    def www_authenticate_uri(self):
        www_authenticate_uri = self._adapter.get_endpoint(interface=plugin.AUTH_INTERFACE)
        if isinstance(self._adapter.auth, _auth.AuthTokenPlugin):
            www_authenticate_uri = urllib.parse.urljoin(www_authenticate_uri, '/').rstrip('/')
        return www_authenticate_uri

    @property
    def auth_version(self):
        return self._request_strategy.AUTH_VERSION

    @property
    def _request_strategy(self):
        if not self._request_strategy_obj:
            strategy_class = self._get_strategy_class()
            self._adapter.version = strategy_class.AUTH_VERSION
            self._request_strategy_obj = strategy_class(self._adapter, include_service_catalog=self._include_service_catalog, requested_auth_interface=self._requested_auth_interface)
        return self._request_strategy_obj

    def _get_strategy_class(self):
        if self._requested_auth_version:
            if not discover.version_match(_V3RequestStrategy.AUTH_VERSION, self._requested_auth_interface):
                self._LOG.info('A version other than v3 was requested: %s', self._requested_auth_interface)
            return _V3RequestStrategy
        for klass in _REQUEST_STRATEGIES:
            if self._adapter.get_endpoint(version=klass.AUTH_VERSION):
                self._LOG.debug('Auth Token confirmed use of %s apis', klass.AUTH_VERSION)
                return klass
        versions = ['v%d.%d' % s.AUTH_VERSION for s in _REQUEST_STRATEGIES]
        self._LOG.error('No attempted versions [%s] supported by server', ', '.join(versions))
        msg = _('No compatible apis supported by server')
        raise ksm_exceptions.ServiceError(msg)

    def verify_token(self, user_token, retry=True, allow_expired=False):
        """Authenticate user token with identity server.

        :param user_token: user's token id
        :param retry: flag that forces the middleware to retry
                      user authentication when an indeterminate
                      response is received. Optional.
        :param allow_expired: Allow retrieving an expired token.
        :returns: access info received from identity server on success
        :rtype: :py:class:`keystoneauth1.access.AccessInfo`
        :raises exc.InvalidToken: if token is rejected
        :raises exc.ServiceError: if unable to authenticate token

        """
        try:
            auth_ref = self._request_strategy.verify_token(user_token, allow_expired=allow_expired)
        except ksa_exceptions.NotFound as e:
            self._LOG.info('Authorization failed for token')
            self._LOG.info('Identity response: %s', e.response.text)
            raise ksm_exceptions.InvalidToken(_('Token authorization failed'))
        except ksa_exceptions.Unauthorized as e:
            self._LOG.info('Identity server rejected authorization')
            self._LOG.warning('Identity response: %s', e.response.text)
            if retry:
                self._LOG.info('Retrying validation')
                return self.verify_token(user_token, False)
            msg = _('Identity server rejected authorization necessary to fetch token data')
            raise ksm_exceptions.ServiceError(msg)
        except ksa_exceptions.HttpError as e:
            self._LOG.error('Bad response code while validating token: %s %s', e.http_status, e.message)
            if hasattr(e.response, 'text'):
                self._LOG.warning('Identity response: %s', e.response.text)
            msg = _('Failed to fetch token data from identity server')
            raise ksm_exceptions.ServiceError(msg)
        else:
            return auth_ref

    def invalidate(self):
        return self._adapter.invalidate()