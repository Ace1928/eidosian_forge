import webob
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import strutils
import requests
Handle incoming request. authenticate and send downstream.