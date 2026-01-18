from unittest import mock
import uuid
from saharaclient.api import base as sahara_base
from heat.common import exception
from heat.engine.clients.os import sahara
from heat.tests import common
from heat.tests import utils
Tests the get_job_type function.