import json
import os
import urllib
from oslo_log import log as logging
import requests
from os_brick import exception
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator.connectors import base
from os_brick.privileged import scaleio as priv_scaleio
from os_brick import utils
def _verify_cert(self):
    verify_cert = self.verify_certificate
    if self.verify_certificate and self.certificate_path:
        verify_cert = self.certificate_path
    return verify_cert