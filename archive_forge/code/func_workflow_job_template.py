from __future__ import absolute_import, division, print_function
import io
import os
import json
import datetime
import importlib
from contextlib import redirect_stdout, suppress
from unittest import mock
import logging
from requests.models import Response, PreparedRequest
import pytest
from ansible.module_utils.six import raise_from
from awx.main.tests.functional.conftest import _request
from awx.main.tests.functional.conftest import credentialtype_scm, credentialtype_ssh  # noqa: F401; pylint: disable=unused-variable
from awx.main.models import (
from django.db import transaction
@pytest.fixture
def workflow_job_template(organization, inventory):
    return WorkflowJobTemplate.objects.create(name='test-workflow_job_template', organization=organization, inventory=inventory)