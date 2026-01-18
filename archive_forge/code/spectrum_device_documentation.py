from __future__ import absolute_import, division, print_function
from socket import gethostbyname, gaierror
import xml.etree.ElementTree as ET
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
Query OneClick for the device using the IP Address