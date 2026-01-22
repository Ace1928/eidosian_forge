import os
import unittest
import time
from boto.swf.layer1 import Layer1
from boto.swf import exceptions as swf_exceptions
class SimpleWorkflowLayer1Test(SimpleWorkflowLayer1TestBase):

    def test_list_domains(self):
        r = self.conn.list_domains('REGISTERED')
        found = None
        for info in r['domainInfos']:
            if info['name'] == self._domain:
                found = info
                break
        self.assertNotEqual(found, None, 'list_domains; test domain not found')
        self.assertEqual(found['description'], self._domain_description, 'list_domains; description does not match')
        self.assertEqual(found['status'], 'REGISTERED', 'list_domains; status does not match')

    def test_list_workflow_types(self):
        r = self.conn.list_workflow_types(self._domain, 'REGISTERED')
        found = None
        for info in r['typeInfos']:
            if info['workflowType']['name'] == self._workflow_type_name and info['workflowType']['version'] == self._workflow_type_version:
                found = info
                break
        self.assertNotEqual(found, None, 'list_workflow_types; test type not found')
        self.assertEqual(found['description'], self._workflow_type_description, 'list_workflow_types; description does not match')
        self.assertEqual(found['status'], 'REGISTERED', 'list_workflow_types; status does not match')

    def test_list_activity_types(self):
        r = self.conn.list_activity_types(self._domain, 'REGISTERED')
        found = None
        for info in r['typeInfos']:
            if info['activityType']['name'] == self._activity_type_name:
                found = info
                break
        self.assertNotEqual(found, None, 'list_activity_types; test type not found')
        self.assertEqual(found['description'], self._activity_type_description, 'list_activity_types; description does not match')
        self.assertEqual(found['status'], 'REGISTERED', 'list_activity_types; status does not match')

    def test_list_closed_workflow_executions(self):
        latest_date = time.time()
        oldest_date = time.time() - 3600
        self.conn.list_closed_workflow_executions(self._domain, start_latest_date=latest_date, start_oldest_date=oldest_date)
        self.conn.list_closed_workflow_executions(self._domain, close_latest_date=latest_date, close_oldest_date=oldest_date)
        self.conn.list_closed_workflow_executions(self._domain, close_latest_date=latest_date, close_oldest_date=oldest_date, close_status='COMPLETED')
        self.conn.list_closed_workflow_executions(self._domain, close_latest_date=latest_date, close_oldest_date=oldest_date, tag='ig')
        self.conn.list_closed_workflow_executions(self._domain, close_latest_date=latest_date, close_oldest_date=oldest_date, workflow_id='ig')
        self.conn.list_closed_workflow_executions(self._domain, close_latest_date=latest_date, close_oldest_date=oldest_date, workflow_name='ig', workflow_version='ig')
        self.conn.list_closed_workflow_executions(self._domain, close_latest_date=latest_date, close_oldest_date=oldest_date, reverse_order=True)

    def test_list_open_workflow_executions(self):
        latest_date = time.time()
        oldest_date = time.time() - 3600
        self.conn.list_closed_workflow_executions(self._domain, latest_date, oldest_date)
        self.conn.list_closed_workflow_executions(self._domain, latest_date, oldest_date, tag='ig')
        self.conn.list_closed_workflow_executions(self._domain, latest_date, oldest_date, workflow_id='ig')
        self.conn.list_closed_workflow_executions(self._domain, latest_date, oldest_date, workflow_name='ig', workflow_version='ig')
        self.conn.list_closed_workflow_executions(self._domain, latest_date, oldest_date, reverse_order=True)