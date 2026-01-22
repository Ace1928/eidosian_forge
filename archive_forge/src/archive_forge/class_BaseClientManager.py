from keystoneauth1 import adapter
from oslo_serialization import jsonutils
import requests
from blazarclient import exception
from blazarclient.i18n import _
class BaseClientManager(object):
    """Base class for managing resources of Blazar."""
    user_agent = 'python-blazarclient'

    def __init__(self, blazar_url, auth_token, session, **kwargs):
        self.blazar_url = blazar_url
        self.auth_token = auth_token
        self.session = session
        if self.session:
            self.request_manager = SessionClient(session=self.session, user_agent=self.user_agent, **kwargs)
        elif self.blazar_url and self.auth_token:
            self.request_manager = RequestManager(blazar_url=self.blazar_url, auth_token=self.auth_token, user_agent=self.user_agent)
        else:
            raise exception.InsufficientAuthInformation