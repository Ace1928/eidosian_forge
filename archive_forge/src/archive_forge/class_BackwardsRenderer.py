import sys
import os
import json
import traceback
import warnings
from io import StringIO, BytesIO
import webob
from webob.exc import HTTPNotFound
from webtest import TestApp
from pecan import (
from pecan.templating import (
from pecan.decorators import accept_noncanonical
from pecan.tests import PecanTestCase
import unittest
class BackwardsRenderer(MakoRenderer):

    def render(self, template_path, namespace):
        namespace = dict(((k, v[::-1]) for k, v in namespace.items()))
        return super(BackwardsRenderer, self).render(template_path, namespace)