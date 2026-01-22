import xml.sax
from boto import handler
from boto.emr import emrobject
from boto.resultset import ResultSet
from tests.compat import unittest
class TestEMRResponses(unittest.TestCase):

    def _parse_xml(self, body, markers):
        rs = ResultSet(markers)
        h = handler.XmlHandler(rs, None)
        xml.sax.parseString(body, h)
        return rs

    def _assert_fields(self, response, **fields):
        for field, expected in fields.items():
            actual = getattr(response, field)
            self.assertEquals(expected, actual, 'Field %s: %r != %r' % (field, expected, actual))

    def test_JobFlows_example(self):
        [jobflow] = self._parse_xml(JOB_FLOW_EXAMPLE, [('member', emrobject.JobFlow)])
        self._assert_fields(jobflow, creationdatetime='2009-01-28T21:49:16Z', startdatetime='2009-01-28T21:49:16Z', state='STARTING', instancecount='4', jobflowid='j-3UN6WX5RRO2AG', loguri='mybucket/subdir/', name='MyJobFlowName', availabilityzone='us-east-1a', slaveinstancetype='m1.small', masterinstancetype='m1.small', ec2keyname='myec2keyname', keepjobflowalivewhennosteps='true')

    def test_JobFlows_completed(self):
        [jobflow] = self._parse_xml(JOB_FLOW_COMPLETED, [('member', emrobject.JobFlow)])
        self._assert_fields(jobflow, creationdatetime='2010-10-21T01:00:25Z', startdatetime='2010-10-21T01:03:59Z', enddatetime='2010-10-21T01:44:18Z', state='COMPLETED', instancecount='10', jobflowid='j-3H3Q13JPFLU22', loguri='s3n://example.emrtest.scripts/jobflow_logs/', name='RealJobFlowName', availabilityzone='us-east-1b', slaveinstancetype='m1.large', masterinstancetype='m1.large', ec2keyname='myubersecurekey', keepjobflowalivewhennosteps='false')
        self.assertEquals(6, len(jobflow.steps))
        self.assertEquals(2, len(jobflow.instancegroups))