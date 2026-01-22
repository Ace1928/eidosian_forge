import base64
import binascii
import datetime
import email.message
import functools
import hashlib
import io
import logging
import os
import random
import re
import socket
import time
import warnings
import weakref
from datetime import datetime as _DatetimeClass
from ipaddress import ip_address
from pathlib import Path
from urllib.request import getproxies, proxy_bypass
import dateutil.parser
from dateutil.tz import tzutc
from urllib3.exceptions import LocationParseError
import botocore
import botocore.awsrequest
import botocore.httpsession
from botocore.compat import HEX_PAT  # noqa: F401
from botocore.compat import IPV4_PAT  # noqa: F401
from botocore.compat import IPV6_ADDRZ_PAT  # noqa: F401
from botocore.compat import IPV6_PAT  # noqa: F401
from botocore.compat import LS32_PAT  # noqa: F401
from botocore.compat import UNRESERVED_PAT  # noqa: F401
from botocore.compat import ZONE_ID_PAT  # noqa: F401
from botocore.compat import (
from botocore.exceptions import (
class IMDSRegionProvider:

    def __init__(self, session, environ=None, fetcher=None):
        """Initialize IMDSRegionProvider.
        :type session: :class:`botocore.session.Session`
        :param session: The session is needed to look up configuration for
            how to contact the instance metadata service. Specifically the
            whether or not it should use the IMDS region at all, and if so how
            to configure the timeout and number of attempts to reach the
            service.
        :type environ: None or dict
        :param environ: A dictionary of environment variables to use. If
            ``None`` is the argument then ``os.environ`` will be used by
            default.
        :type fecther: :class:`botocore.utils.InstanceMetadataRegionFetcher`
        :param fetcher: The class to actually handle the fetching of the region
            from the IMDS. If not provided a default one will be created.
        """
        self._session = session
        if environ is None:
            environ = os.environ
        self._environ = environ
        self._fetcher = fetcher

    def provide(self):
        """Provide the region value from IMDS."""
        instance_region = self._get_instance_metadata_region()
        return instance_region

    def _get_instance_metadata_region(self):
        fetcher = self._get_fetcher()
        region = fetcher.retrieve_region()
        return region

    def _get_fetcher(self):
        if self._fetcher is None:
            self._fetcher = self._create_fetcher()
        return self._fetcher

    def _create_fetcher(self):
        metadata_timeout = self._session.get_config_variable('metadata_service_timeout')
        metadata_num_attempts = self._session.get_config_variable('metadata_service_num_attempts')
        imds_config = {'ec2_metadata_service_endpoint': self._session.get_config_variable('ec2_metadata_service_endpoint'), 'ec2_metadata_service_endpoint_mode': resolve_imds_endpoint_mode(self._session), 'ec2_metadata_v1_disabled': self._session.get_config_variable('ec2_metadata_v1_disabled')}
        fetcher = InstanceMetadataRegionFetcher(timeout=metadata_timeout, num_attempts=metadata_num_attempts, env=self._environ, user_agent=self._session.user_agent(), config=imds_config)
        return fetcher