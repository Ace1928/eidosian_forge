import time
from unittest import mock
import uuid
from pycadf import attachment
from pycadf import cadftype
from pycadf import credential
from pycadf import endpoint
from pycadf import event
from pycadf import geolocation
from pycadf import host
from pycadf import identifier
from pycadf import measurement
from pycadf import metric
from pycadf import reason
from pycadf import reporterstep
from pycadf import resource
from pycadf import tag
from pycadf.tests import base
from pycadf import timestamp
def test_reporterstep(self):
    step = reporterstep.Reporterstep(role='modifier', reporter=resource.Resource(typeURI='storage'), reporterId=identifier.generate_uuid(), reporterTime=timestamp.get_utc_now())
    self.assertEqual(False, step.is_valid())
    dict_step = step.as_dict()
    for key in reporterstep.REPORTERSTEP_KEYNAMES:
        self.assertIn(key, dict_step)
    step = reporterstep.Reporterstep(role='modifier', reporter=resource.Resource(typeURI='storage'), reporterTime=timestamp.get_utc_now())
    self.assertEqual(True, step.is_valid())
    step = reporterstep.Reporterstep(role='modifier', reporterId=identifier.generate_uuid(), reporterTime=timestamp.get_utc_now())
    self.assertEqual(True, step.is_valid())