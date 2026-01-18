from packaging.version import Version
from requests.adapters import HTTPAdapter
from docker.transport.basehttpadapter import BaseHTTPAdapter
import urllib3

        Ensure assert_hostname is set correctly on our pool

        We already take care of a normal poolmanager via init_poolmanager

        But we still need to take care of when there is a proxy poolmanager
        