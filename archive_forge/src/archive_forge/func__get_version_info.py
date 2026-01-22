import logging
import urllib.parse as urlparse
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient import httpclient
from keystoneclient.i18n import _
@staticmethod
def _get_version_info(version, root_url):
    """Parse version information.

        :param version: a dict of a Keystone version response
        :param root_url: string url used to construct
            the version if no URL is provided.
        :returns: tuple - (verionId, versionStatus, versionUrl)
        """
    id = version['id']
    status = version['status']
    ref = urlparse.urljoin(root_url, id)
    if 'links' in version:
        for link in version['links']:
            if link['rel'] == 'self':
                ref = link['href']
                break
    return (id, status, ref)