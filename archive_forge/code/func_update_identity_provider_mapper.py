from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def update_identity_provider_mapper(self, mapper, alias, realm='master'):
    """ Update an existing identity provider.
        :param mapper: IdentityProviderMapperRepresentation of the mapper to be updated.
        :param alias: Alias of the identity provider.
        :param realm: Realm in which this identity provider resides, default "master".
        :return HTTPResponse object on success
        """
    mapper_url = URL_IDENTITY_PROVIDER_MAPPER.format(url=self.baseurl, realm=realm, alias=alias, id=mapper['id'])
    try:
        return open_url(mapper_url, method='PUT', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, data=json.dumps(mapper), validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Could not update mapper %s for identity provider %s in realm %s: %s' % (mapper['id'], alias, realm, str(e)))