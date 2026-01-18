import http.client as http
import os
import re
import time
import psutil
import requests
from glance.tests import functional
from glance.tests.utils import execute
Test SIGHUP picks up new config values