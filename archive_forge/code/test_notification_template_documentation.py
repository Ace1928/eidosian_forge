from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import NotificationTemplate, Job
Job notification templates may encounter undefined values in the context when they are
    rendered. Make sure that accessing attributes or items of an undefined value returns another
    instance of Undefined, rather than raising an UndefinedError. This enables the use of expressions
    like "{{ job.created_by.first_name | default('unknown') }}".